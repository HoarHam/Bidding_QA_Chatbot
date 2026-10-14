"""ORM 基类与命名约定。

## 为什么要显式约定约束名

不指定名字时，数据库会自己生成（Postgres 用 `users_username_key` 这类），
而 Alembic 要删除或修改约束时必须写出确切名字。名字由数据库随机决定的话，
自动生成的迁移会在不同环境产生不同 SQL。约定死名字可以避免这个坑。
"""

from __future__ import annotations

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
