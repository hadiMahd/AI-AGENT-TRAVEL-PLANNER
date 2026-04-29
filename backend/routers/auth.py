"""
Auth endpoints — signup and login.

Signup: creates a new user with a bcrypt-hashed password, returns a JWT.
Login:  verifies credentials against the stored hash, returns a JWT.

Both endpoints return the same TokenResponse — the frontend stores the
access_token in localStorage and sends it as "Authorization: Bearer <token>"
on all subsequent requests.

Why hash on signup but verify on login?
- Signup: we receive plaintext, hash it, store the hash — plaintext is discarded
- Login:  we receive plaintext, compare against the stored hash — plaintext is discarded
- The database NEVER stores plaintext passwords — only bcrypt hashes

Why return a JWT immediately on signup?
- Better UX — the user is signed up AND logged in with one request
- No redirect to a separate login page after registration
- The frontend just stores the token and navigates to the dashboard

Why 409 for duplicate email instead of 400?
- 409 Conflict specifically means "resource already exists"
- 400 Bad Request is too generic — it could mean malformed JSON, missing fields, etc.
- 409 tells the frontend exactly what happened: "that email is taken"
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_db
from dependencies import CurrentUserDep
from models.alchemy import User
from models.schemas import LoginRequest, SignupRequest, TokenResponse
from services.auth import create_access_token, hash_password, verify_password

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post(
	"/signup",
	response_model=TokenResponse,
	status_code=status.HTTP_201_CREATED,
	summary="Create a new account and get a JWT",
	description=(
		"Registers a new user with email + password. "
		"Password is bcrypt-hashed before storage. "
		"Returns a JWT access token so the user is immediately logged in."
	),
)
async def signup(
	body: SignupRequest,
	db: AsyncSession = Depends(get_db),
) -> TokenResponse:
	"""
	Create a new user account.

	Steps:
	1. Check if the email is already registered → 409 if taken
	2. Hash the password with bcrypt
	3. Insert a new User row into the database
	4. Create a JWT with the user's ID as the "sub" claim
	5. Return the token — user is immediately authenticated
	"""
	# Step 1: Check for existing user with the same email
	result = await db.execute(select(User).where(User.email == body.email))
	existing = result.scalar_one_or_none()
	if existing is not None:
		logger.warning("Signup failed — email already registered: %s", body.email)
		raise HTTPException(
			status_code=status.HTTP_409_CONFLICT,
			detail="A user with this email already exists",
		)

	# Step 2: Hash the password — never store plaintext
	hashed = hash_password(body.password)

	# Step 3: Create the user row
	user = User(
		email=body.email,
		password_hash=hashed,
	)
	db.add(user)
	await db.commit()
	# Refresh to get the server-generated id and created_at
	await db.refresh(user)

	logger.info("User created: id=%s, email=%s", user.id, user.email)

	# Step 4: Create JWT with user ID as the subject claim
	access_token = create_access_token(data={"sub": user.id})

	return TokenResponse(access_token=access_token, token_type="bearer")


@router.post(
	"/login",
	response_model=TokenResponse,
	summary="Authenticate and get a JWT",
	description=(
		"Verifies email + password credentials. "
		"Password is verified against the stored bcrypt hash. "
		"Returns a JWT access token on success."
	),
)
async def login(
	body: LoginRequest,
	db: AsyncSession = Depends(get_db),
) -> TokenResponse:
	"""
	Authenticate an existing user.

	Steps:
	1. Look up the user by email → 401 if not found
	2. Verify the password against the stored bcrypt hash → 401 if mismatch
	3. Create a JWT with the user's ID as the "sub" claim
	4. Return the token

	Why 401 for both "user not found" and "wrong password"?
	- Security best practice: don't reveal whether an email is registered
	- "Invalid credentials" covers both cases without leaking info
	- Prevents enumeration attacks (checking which emails exist)
	"""
	# Step 1: Look up user by email
	result = await db.execute(select(User).where(User.email == body.email))
	user = result.scalar_one_or_none()

	# Step 2: Verify password — also covers "user not found" case
	# If user is None, verify_password will return False (any string != None)
	if user is None or not verify_password(body.password, user.password_hash):
		logger.warning("Login failed for email: %s", body.email)
		raise HTTPException(
			status_code=status.HTTP_401_UNAUTHORIZED,
			detail="Invalid credentials",
			headers={"WWW-Authenticate": "Bearer"},
		)

	# Step 3: Create JWT
	access_token = create_access_token(data={"sub": user.id})

	logger.info("User logged in: id=%s, email=%s", user.id, user.email)

	return TokenResponse(access_token=access_token, token_type="bearer")
