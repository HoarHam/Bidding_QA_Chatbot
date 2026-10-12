/**
 * 前端 UI 冒烟验证（真实浏览器）。
 *
 * 为什么必须有这个脚本：本项目已经栽过两次——
 *
 * 1. **Next 16 拦截跨域开发资源**，导致客户端 JS 不接管，所有交互失效；
 *    而服务端渲染的 HTML 完全正常，curl 和静态检查都看不出来。
 * 2. **`rewrites` 代理缓冲 SSE**，答案憋到最后一次性显示；
 *    后端直连是好的，只有真浏览器渲染才能确认「逐字出现」。
 *
 * 这两个问题都只在真浏览器里暴露，所以别再只靠 curl 验证前端。
 *
 * 用法（需先启动后端 :8001 与前端 :3000）：
 *     npm run verify:ui
 *
 * 浏览器：优先用系统自带的 Edge/Chrome（`channel`），避免下载 Chromium。
 * 国内网络下载 Chromium 常常卡死，而 Windows 一定自带 Edge。
 * 都找不到时以退出码 2 结束（表示「跑不了」，区别于「跑失败」）。
 */

import { chromium } from "playwright";

const SITE = process.env.SITE_URL ?? "http://127.0.0.1:3000";
const BACKEND = process.env.API_ORIGIN ?? "http://127.0.0.1:8001";

const results = [];
const consoleErrors = [];

function check(name, ok, detail = "") {
  results.push({ name, ok });
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${name}${detail ? "  — " + detail : ""}`);
}

/** 依次尝试系统浏览器，最后才退回 Playwright 自带的 Chromium。 */
async function launch() {
  const attempts = [
    { channel: "msedge", label: "系统 Edge" },
    { channel: "chrome", label: "系统 Chrome" },
    { label: "Playwright Chromium" },
  ];
  const errors = [];
  for (const a of attempts) {
    try {
      const opts = a.channel ? { channel: a.channel, headless: true } : { headless: true };
      const browser = await chromium.launch(opts);
      console.log(`  浏览器: ${a.label}\n`);
      return browser;
    } catch (e) {
      errors.push(`${a.label}: ${e.message.split("\n")[0]}`);
    }
  }
  console.error("  找不到可用浏览器，跳过 UI 验证：");
  errors.forEach((e) => console.error("    - " + e));
  process.exit(2);
}

/**
 * 采样最后一条助手消息的文本长度随时间的变化。
 *
 * 判据用「是否存在**中间态**样本」（长度既非 0 也非最终值），
 * 而不是「间隔大小」——token 生成本来就快，网络还会合并小包，
 * 拿间隔判断会把正常的流式误判成缓冲。
 */
async function sampleAssistantGrowth(page, timeoutMs = 90000) {
  const samples = [];
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    const s = await page.evaluate(() => {
      const els = document.querySelectorAll('[data-role="assistant"]');
      const el = els[els.length - 1];
      if (!el) return null;
      return { len: (el.innerText || "").length, streaming: el.dataset.streaming === "true" };
    });
    if (s) {
      samples.push(s);
      if (!s.streaming && s.len > 0) break; // 流结束
    }
    await page.waitForTimeout(120);
  }
  return samples;
}

async function main() {
  // 先确认后端在跑，否则「发送成功但没回答」会污染判断
  try {
    const r = await fetch(`${BACKEND}/api/health`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
  } catch (e) {
    console.error(`  后端未就绪（${BACKEND}）：${e.message}`);
    console.error("  请先运行：uv run bidding-qa-chatbot serve");
    process.exit(2);
  }

  const browser = await launch();
  const page = await browser.newPage();
  page.on("console", (m) => {
    if (m.type() === "error") consoleErrors.push(m.text());
  });
  page.on("pageerror", (e) => consoleErrors.push("PAGEERROR: " + e.message));

  await page.goto(SITE, { waitUntil: "load" });
  await page.waitForTimeout(2500); // 等 hydration

  // ---------- 交互可用性（对应「客户端 JS 未接管」那类故障）----------
  const thinking = page.getByRole("button", { name: "深度思考" });
  const fast = page.getByRole("button", { name: "快速" });
  const before = await thinking.getAttribute("aria-pressed");
  await thinking.click();
  await page.waitForTimeout(300);
  const afterThinking = await thinking.getAttribute("aria-pressed");
  const afterFast = await fast.getAttribute("aria-pressed");
  check(
    "模式切换响应点击",
    before !== afterThinking && afterThinking === "true" && afterFast === "false",
    `深度思考 ${before} -> ${afterThinking}`,
  );
  await fast.click();
  await page.waitForTimeout(200);

  // ---------- 示例问题：真实文本、可点击填入 ----------
  const chip = page.getByRole("button", { name: /单一来源异议被驳回/ });
  check("示例问题存在", (await chip.count()) === 1);
  await chip.click();
  await page.waitForTimeout(300);
  const filled = await page.locator("textarea").inputValue();
  check("点击示例填入输入框", filled.includes("单一来源异议被驳回"), `"${filled.slice(0, 20)}…"`);
  await page.locator("textarea").fill("");

  // ---------- Enter 发送 ----------
  await page.locator("textarea").type("这是Enter发送测试", { delay: 20 });
  await page.locator("textarea").press("Enter");
  await page.waitForTimeout(1500);
  const afterEnter = await page.locator("textarea").inputValue();
  const bodyText = await page.locator("body").innerText();
  check(
    "Enter 触发发送",
    afterEnter === "" && bodyText.includes("这是Enter发送测试"),
    `输入框清空=${afterEnter === ""}`,
  );

  // ---------- Shift+Enter 换行 ----------
  await page.locator("textarea").fill("");
  await page.locator("textarea").type("换行测试");
  await page.locator("textarea").press("Shift+Enter");
  await page.waitForTimeout(400);
  check("Shift+Enter 换行不发送", (await page.locator("textarea").inputValue()).includes("\n"));
  await page.locator("textarea").fill("");

  // ---------- 只有一个发送按钮 ----------
  await page
    .getByRole("button", { name: "停止" })
    .waitFor({ state: "detached", timeout: 60000 })
    .catch(() => {});
  const sendCount = await page.getByRole("button", { name: "发送", exact: true }).count();
  check("页面上只有 1 个「发送」按钮", sendCount === 1, `实际 ${sendCount} 个`);

  // ---------- 流式渲染（本次新增的核心检查）----------
  await page.locator("textarea").type("请简要说明投标保证金的退还规定。", { delay: 10 });
  await page.getByRole("button", { name: "发送", exact: true }).click();

  const samples = await sampleAssistantGrowth(page);
  const finalLen = samples.length ? Math.max(...samples.map((s) => s.len)) : 0;
  const intermediates = samples.filter((s) => s.len > 0 && s.len < finalLen).length;
  check(
    "回答逐字流式渲染（存在中间态，非一次性出现）",
    finalLen > 20 && intermediates > 0,
    `最终 ${finalLen} 字，中间态样本 ${intermediates} 个，采样 ${samples.length} 次`,
  );

  // ---------- Agent 工具调用 ----------
  // 招投标问题应当触发知识库检索，且过程要可见——否则用户会以为界面卡住。
  const toolItems = await page.getByTestId("tool-activity").count();
  const toolText = toolItems ? await page.getByTestId("tool-activity").first().innerText() : "";
  check(
    "Agent 调用知识库检索且过程可见",
    toolItems > 0 && /检索/.test(toolText),
    toolItems > 0 ? `"${toolText.replace(/\s+/g, " ").slice(0, 46)}"` : "未出现工具调用条目",
  );

  // 参考来源面板应在检索后填充（Agent 路径下来源由 tool_result 带出）
  const sourceCards = await page.locator("aside [class*='rounded-lg']").count();
  check("参考来源面板已填充", sourceCards > 0, `${sourceCards} 条`);

  // ---------- 控制台无错误 ----------
  const runtimeErrors = consoleErrors.filter((e) =>
    /hydrat|Minified React error|Cannot read|is not a function|undefined|Encountered two children/i.test(
      e,
    ),
  );
  check(
    "无 hydration / React 运行时错误",
    runtimeErrors.length === 0,
    runtimeErrors[0]?.slice(0, 140) ?? "无",
  );

  const otherErrors = consoleErrors.filter((e) => !runtimeErrors.includes(e));
  check("控制台无其他报错", otherErrors.length === 0, otherErrors[0]?.slice(0, 140) ?? "无");

  await page.screenshot({ path: "ui-verify.png", fullPage: true }).catch(() => {});
  await browser.close();

  const failed = results.filter((r) => !r.ok);
  console.log(`\n  ==== ${results.length - failed.length}/${results.length} 通过 ====`);
  process.exit(failed.length ? 1 : 0);
}

main().catch((e) => {
  console.error("  脚本异常:", e.message);
  process.exit(2);
});
