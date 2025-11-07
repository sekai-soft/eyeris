from sqlalchemy import Column, Integer, String, ForeignKey, UUID, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
import uuid as uuid_lib

Base = declarative_base()


class Profile(Base):
    """
    Profile model representing a user profile.

    Each profile has a unique identifier and can have multiple embeddings
    associated with it.
    """
    __tablename__ = "profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(UUID(as_uuid=True), unique=True, nullable=False, default=uuid_lib.uuid4)
    profile_id = Column(String, unique=True, nullable=False, index=True)  # User-facing identifier
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationship to embeddings
    embeddings = relationship("Embedding", back_populates="profile", cascade="all, delete-orphan")


class Embedding(Base):
    """
    Embedding model representing a vector embedding associated with a profile.

    Each embedding is linked to an image URL and stores the actual vector
    representation.
    """
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(UUID(as_uuid=True), unique=True, nullable=False, default=uuid_lib.uuid4)
    profile_uuid = Column(UUID(as_uuid=True), ForeignKey("profiles.uuid", ondelete="CASCADE"), nullable=False)
    image_url = Column(String, nullable=False)
    # ResNet-50 outputs 2048-dimensional feature vectors
    embedding_vector = Column(Vector(2048), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationship back to profile
    profile = relationship("Profile", back_populates="embeddings")
