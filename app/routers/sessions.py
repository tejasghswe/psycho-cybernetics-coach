from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request

from app.graph import get_session, resume_session, start_session
from schemas import AnswerRequest, CreateSessionRequest, SessionResponse, SessionStatus

router = APIRouter()


@router.post("/sessions", response_model=SessionResponse)
async def create_session(payload: CreateSessionRequest, request: Request) -> SessionResponse:
    session_id = str(uuid.uuid4())
    graph = request.app.state.graph
    return await start_session(graph, session_id, payload.mood_selection.value, payload.transcript)


@router.post("/sessions/{session_id}/answer", response_model=SessionResponse)
async def answer_session(session_id: str, payload: AnswerRequest, request: Request) -> SessionResponse:
    graph = request.app.state.graph
    current = await get_session(graph, session_id)
    if current is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    if current.status != SessionStatus.ACTIVE or current.question is None:
        raise HTTPException(status_code=409, detail="Session has no pending question to answer.")

    response = await resume_session(graph, session_id, payload.answer)
    if response is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return response


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session_route(session_id: str, request: Request) -> SessionResponse:
    graph = request.app.state.graph
    response = await get_session(graph, session_id)
    if response is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return response
