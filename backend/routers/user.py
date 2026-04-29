"""
User endpoints — profile and history (scoped to authenticated user).

All endpoints here require Depends(get_current_user) — every query
and response is scoped to the authenticated user. No cross-user access.

Per INSTRUCTIONS.md: "Scope all runs, history, webhook targets to
authenticated user."
"""

import logging

from fastapi import APIRouter, Depends
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_db
from dependencies import CurrentUserDep
from models.alchemy import AgentRun, User
from models.schemas import UserOut

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/user", tags=["User"])


@router.get(
	"/me",
	response_model=UserOut,
	summary="Get the current authenticated user's profile",
	description="Returns the authenticated user's ID, email, and registration date.",
)
async def get_my_profile(user: User = CurrentUserDep) -> UserOut:
	"""
	Return the current user's profile.

	The user object comes from Depends(get_current_user) which:
	1. Extracts the JWT from the Authorization header
	2. Validates the token signature and expiration
	3. Looks up the User row in the database
	4. Returns the ORM object (or raises 401)

	We just need to serialize it to the UserOut Pydantic schema.
	"""
	return UserOut.model_validate(user)


@router.get(
	"/stats",
	summary="Get usage stats for the current user",
	description="Returns the number of agent runs and other usage metrics.",
)
async def get_my_stats(
	user: User = CurrentUserDep,
	db: AsyncSession = Depends(get_db),
) -> dict:
	"""
	Return usage statistics scoped to the current user.

	Returns the count of agent runs made by this user.
	More stats can be added as the system grows.
	"""

	result = await db.execute(
		select(func.count()).where(AgentRun.user_id == user.id)
	)
	run_count = result.scalar() or 0

	return {
		"email": user.email,
		"agent_runs": run_count,
	}
