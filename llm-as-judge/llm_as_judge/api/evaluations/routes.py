"""
FastAPI routes for LLM Judge evaluation endpoints
Exposes evaluation functionality via REST API
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, status, Query, Depends, File, UploadFile
from fastapi.responses import JSONResponse

from llm_as_judge import injector
from llm_as_judge.api.evaluations.services import IEvaluationService
from llm_as_judge.api.evaluations.schemas import (
    SingleEvaluationRequest,
    SingleEvaluationResponse,
    HallucinationEvaluationRequest,
    QAEvaluationRequest,
    ToxicityEvaluationRequest,
    BatchEvaluationRequest,
    BatchEvaluationResponse,
    HealthCheckResponse,
    EvaluationMetricEnum
)

logger = logging.getLogger(__name__)


def get_evaluation_service() -> IEvaluationService:
    """
    Dependency injection for EvaluationService
    Returns the singleton service instance
    """
    return injector.get(IEvaluationService)


# Create router
router = APIRouter(
    prefix="/api/v1/evaluations",
    tags=["evaluations"],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"}
    }
)


# ============================================================================
# Health Check Endpoints
# ============================================================================

@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health check endpoint",
    description="Check if the evaluation service is running"
)
async def health_check(
        service: IEvaluationService = Depends(get_evaluation_service)
) -> HealthCheckResponse:
    """
    Health check endpoint

    Returns service status and configuration info
    """
    try:
        config = service.get_service_config()
        return HealthCheckResponse(
            status="healthy",
            version="1.0.0",
            service="LLM Judge Evaluation API"
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Service health check failed"
        )


@router.get(
    "/config",
    summary="Get service configuration",
    description="Retrieve current service configuration"
)
async def get_config(
        service: IEvaluationService = Depends(get_evaluation_service)
):
    """Get current service configuration"""
    try:
        return service.get_service_config()
    except Exception as e:
        logger.error(f"Error getting config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve configuration"
        )


# ============================================================================
# Single Response Evaluation Endpoints
# ============================================================================

@router.post(
    "/evaluate",
    response_model=SingleEvaluationResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate agent response",
    description="Evaluate a single agent response across multiple metrics"
)
async def evaluate_agent_response(
        request: SingleEvaluationRequest,
        service: IEvaluationService = Depends(get_evaluation_service)
) -> SingleEvaluationResponse:
    """
    Evaluate a single agent response

    Evaluates the given agent output against specified metrics:
    - hallucination: Detects if output contains made-up information
    - qa: Evaluates answer correctness
    - toxicity: Detects harmful or biased content
    - relevance: Checks document relevance

    Args:
        request: SingleEvaluationRequest with agent output and metrics
        service: Injected EvaluationService

    Returns:
        SingleEvaluationResponse with evaluation results

    Raises:
        HTTPException: If evaluation fails
    """
    try:
        logger.info(f"Received evaluation request for agent: {request.agent_id}")
        response = service.evaluate_agent_response(request)
        return response

    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error evaluating agent response: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to evaluate agent response"
        )


# ============================================================================
# Specific Metric Evaluation Endpoints
# ============================================================================

@router.post(
    "/hallucination",
    response_model=dict,
    summary="Evaluate hallucination",
    description="Check if output contains hallucinated information"
)
async def evaluate_hallucination(
        request: HallucinationEvaluationRequest,
        service: IEvaluationService = Depends(get_evaluation_service)
):
    """
    Evaluate hallucination in response

    Determines if the agent output contains factual information not present
    in the reference context

    Args:
        request: HallucinationEvaluationRequest
        service: Injected EvaluationService

    Returns:
        Evaluation result with score and label
    """
    try:
        logger.info("Received hallucination evaluation request")
        response = service.evaluate_hallucination(request)
        return response.dict()

    except Exception as e:
        logger.error(f"Error evaluating hallucination: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to evaluate hallucination"
        )


@router.post(
    "/qa",
    response_model=dict,
    summary="Evaluate Q&A correctness",
    description="Check if answer correctly answers the question"
)
async def evaluate_qa(
        request: QAEvaluationRequest,
        service: IEvaluationService = Depends(get_evaluation_service)
):
    """
    Evaluate Q&A correctness

    Determines if the agent's answer correctly answers the given question

    Args:
        request: QAEvaluationRequest
        service: Injected EvaluationService

    Returns:
        Evaluation result with score and label
    """
    try:
        logger.info("Received Q&A evaluation request")
        response = service.evaluate_qa(request)
        return response.dict()

    except Exception as e:
        logger.error(f"Error evaluating Q&A: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to evaluate Q&A"
        )


@router.post(
    "/toxicity",
    response_model=dict,
    summary="Evaluate toxicity",
    description="Check if text contains harmful or biased content"
)
async def evaluate_toxicity(
        request: ToxicityEvaluationRequest,
        service: IEvaluationService = Depends(get_evaluation_service)
):
    """
    Evaluate toxicity in text

    Detects if the text contains toxic, biased, or harmful content

    Args:
        request: ToxicityEvaluationRequest
        service: Injected EvaluationService

    Returns:
        Evaluation result with score and label
    """
    try:
        logger.info("Received toxicity evaluation request")
        response = service.evaluate_toxicity(request)
        return response.dict()

    except Exception as e:
        logger.error(f"Error evaluating toxicity: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to evaluate toxicity"
        )


# ============================================================================
# Batch Evaluation Endpoints
# ============================================================================

@router.post(
    "/batch",
    response_model=BatchEvaluationResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch evaluate outputs",
    description="Evaluate multiple agent outputs in a single request"
)
async def batch_evaluate(
        request: BatchEvaluationRequest,
        service: IEvaluationService = Depends(get_evaluation_service)
) -> BatchEvaluationResponse:
    """
    Batch evaluate multiple agent outputs

    Efficiently evaluate multiple outputs using the same metric type

    Args:
        request: BatchEvaluationRequest with data and eval type
        service: Injected EvaluationService

    Returns:
        BatchEvaluationResponse with results and summary stats
    """
    try:
        logger.info(f"Received batch evaluation request for {len(request.data)} items")

        if len(request.data) == 0:
            raise ValueError("Data list cannot be empty")

        if len(request.data) > 1000:
            raise ValueError("Batch size limited to 1000 items")

        response = service.batch_evaluate(request)
        return response

    except ValueError as e:
        logger.warning(f"Validation error in batch evaluation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error in batch evaluation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to evaluate batch"
        )


# ============================================================================
# Comparison Endpoints
# ============================================================================

@router.post(
    "/compare",
    response_model=dict,
    summary="Compare agent versions",
    description="Compare two agent versions on the same query"
)
async def compare_agents(
        agent1_id: str = Query(..., description="First agent ID"),
        agent2_id: str = Query(..., description="Second agent ID"),
        query: str = Query(..., description="Query to evaluate"),
        output1: str = Query(..., description="First agent output"),
        output2: str = Query(..., description="Second agent output"),
        reference: Optional[str] = Query(None, description="Reference data"),
        service: IEvaluationService = Depends(get_evaluation_service)
):
    """
    Compare two agent versions

    Evaluates two agents on the same query and compares results
    """
    try:
        logger.info(f"Comparing agents {agent1_id} vs {agent2_id}")
        # Evaluate both agents
        request1 = SingleEvaluationRequest(
            agent_id=agent1_id,
            query=query,
            agent_output=output1,
            reference_data={"reference": reference} if reference else None,
            eval_metrics=[EvaluationMetricEnum.HALLUCINATION, EvaluationMetricEnum.TOXICITY]
        )

        request2 = SingleEvaluationRequest(
            agent_id=agent2_id,
            query=query,
            agent_output=output2,
            reference_data={"reference": reference} if reference else None,
            eval_metrics=[EvaluationMetricEnum.HALLUCINATION, EvaluationMetricEnum.TOXICITY]
        )

        result1 = service.evaluate_agent_response(request1)
        result2 = service.evaluate_agent_response(request2)

        # Calculate winner
        hallucination_score1 = result1.evaluations.get("hallucination", {}).score if "hallucination" in result1.evaluations else 0
        hallucination_score2 = result2.evaluations.get("hallucination", {}).score if "hallucination" in result2.evaluations else 0

        winner = agent1_id if hallucination_score1 > hallucination_score2 else agent2_id if hallucination_score2 > hallucination_score1 else "tie"

        return {
            "query": query,
            f"{agent1_id}_result": result1.dict(),
            f"{agent2_id}_result": result2.dict(),
            "winner": winner,
            "reason": "Based on hallucination detection score"
        }

    except Exception as e:
        logger.error(f"Error comparing agents: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compare agents"
        )


# ============================================================================
# KG Ingestion Endpoints
# ============================================================================

@router.post(
    "/upload",
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest knowledge",
    description="Upload a Markdown file to parse and ingest into the Knowledge Graph"
)
async def upload_document(
        file: UploadFile = File(...),
        service: IEvaluationService = Depends(get_evaluation_service)
):
    """
    Upload and ingest a document
    
    Parses the uploaded Markdown file, extracts entities and triples,
    and stores them in the Neo4j Knowledge Graph.
    
    Args:
        file: The Markdown file to upload
        service: Injected EvaluationService
        
    Returns:
        Summary of extraction and ingestion results
    """
    try:
        logger.info(f"Received upload request for file: {file.filename}")
        
        if not file.filename.endswith(".md"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only .md files are supported for parsing"
            )
            
        content = await file.read()
        text_content = content.decode("utf-8")
        
        result = service.ingest_markdown_file(file.filename, text_content)
        return result

    except ValueError as e:
        logger.warning(f"Validation error in upload: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process document upload"
        )


@router.post(
    "/evaluate-and-correct",
    status_code=status.HTTP_200_OK,
    summary="Evaluate and self-correct response",
    description="Evaluate an agent response against the KG and automatically correct it if unsupported"
)
async def evaluate_and_correct(
        request: SingleEvaluationRequest,
        service: IEvaluationService = Depends(get_evaluation_service)
):
    """
    Evaluate and self-correct a response
    
    1. Retrieval: Pulls relevant facts from Neo4j
    2. Evaluation: Judge checks if response is factual
    3. Repair: If unsupported, Judge rewrites response using KG ground truth
    
    Returns:
        Original output, corrected output (if any), and evaluation details
    """
    try:
        logger.info(f"Received evaluate-and-correct request for agent: {request.agent_id}")
        result = service.evaluate_and_correct(request)
        return result

    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error in evaluate-and-correct: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to evaluate and correct response"
        )


# ============================================================================
# Help Endpoints
# ============================================================================

@router.get(
    "/metrics",
    summary="Get evaluation metrics",
    description="Get list of available evaluation metrics"
)
async def get_metrics():
    """Get available evaluation metrics"""
    return {
        "metrics": [
            {
                "name": "hallucination",
                "description": "Detects if output contains made-up information",
                "labels": ["factual", "hallucinated"]
            },
            {
                "name": "qa",
                "description": "Evaluates answer correctness",
                "labels": ["correct", "incorrect"]
            },
            {
                "name": "toxicity",
                "description": "Detects harmful or biased content",
                "labels": ["toxic", "non-toxic"]
            },
            {
                "name": "relevance",
                "description": "Checks document relevance to query",
                "labels": ["relevant", "unrelated"]
            },
            {
                "name": "kg_verification",
                "description": "Verifies facts using Knowledge Graph",
                "labels": ["factual", "unsupported"]
            }
        ]
    }


@router.get(
    "",
    summary="API documentation",
    description="Get API documentation and available endpoints"
)
async def api_docs():
    """API documentation"""
    return {
        "service": "LLM Judge Evaluation API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/v1/evaluations/health",
            "config": "/api/v1/evaluations/config",
            "metrics": "/api/v1/evaluations/metrics",
            "evaluate": "/api/v1/evaluations/evaluate (POST)",
            "upload": "/api/v1/evaluations/upload (POST)",
            "hallucination": "/api/v1/evaluations/hallucination (POST)",
            "qa": "/api/v1/evaluations/qa (POST)",
            "toxicity": "/api/v1/evaluations/toxicity (POST)",
            "batch": "/api/v1/evaluations/batch (POST)",
            "compare": "/api/v1/evaluations/compare (POST)"
        }
    }