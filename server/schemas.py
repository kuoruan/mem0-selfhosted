from pydantic import BaseModel


class MessageResponse(BaseModel):
    message: str


class UserInfo(BaseModel):
    """Minimal user reference for embedding in other responses (e.g. permission grantee/grantor)."""

    id: str
    name: str
    email: str
