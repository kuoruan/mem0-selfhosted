from typing import Optional

from auth import require_admin
from db import get_db
from fastapi import APIRouter, Depends, Query, Request
from models import User
from pydantic import BaseModel
from routers.auth import UserResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from utils.pagination import paginate_response

router = APIRouter(prefix="/users", tags=["users"])


class UserListResponse(BaseModel):
    """Paginated envelope mirroring ``utils.pagination.paginate_response``."""

    count: int
    next: Optional[str] = None
    previous: Optional[str] = None
    results: list[UserResponse]


@router.get("", response_model=UserListResponse, summary="List dashboard users (admin only)")
def list_users(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=200),
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Return all dashboard users for the admin grant/transfer user picker.

    Admin-only (``require_admin``). Reuses ``UserResponse`` so the item shape matches
    ``/auth/me``. Pagination is done at the DB level (``LIMIT``/``OFFSET`` + a
    ``COUNT`` for the total) to avoid loading every user row into memory.
    """
    total = db.scalar(select(func.count(User.id))) or 0
    users = (
        db.execute(select(User).order_by(User.name.asc()).offset((page - 1) * page_size).limit(page_size))
        .scalars()
        .all()
    )
    return paginate_response(
        request,
        users,
        page,
        page_size,
        total=total,
    )
