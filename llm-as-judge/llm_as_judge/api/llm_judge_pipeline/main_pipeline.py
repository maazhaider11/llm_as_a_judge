"""
Base LLM-as-a-Judge Pipeline for Agent Evaluation using Arize Phoenix
This pipeline provides a scalable framework for evaluating other agents using LLM judges.
"""
from typing import Dict, List, Any, Optional
import pandas as pd
from dataclasses import dataclass
from issm_api_common.config.settings import config as app_config
from phoenix.evals.llm import LLM
from phoenix.evals.metrics import (
    HallucinationEvaluator,
    # RelevanceEvaluator,
    # ToxicityEvaluator,
)
from phoenix.evals import llm_classify
from string import Template


@dataclass
class EvaluationConfig:
    """Configuration for the judge evaluation pipeline"""
    model_name: str = "mistral-large-latest"
    provider: str = "mistral"
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 1024
    concurrency: int = 20
    timeout: int = 30


class JudgeEvaluator:
    """Base LLM Judge Evaluator for agent outputs"""

    def __init__(self, config: EvaluationConfig = None):
        """
        Initialize the judge evaluator

        Args:
            config: EvaluationConfig object with model settings
        """
        self.config = config or EvaluationConfig()
        self.api_key = self.config.api_key or getattr(app_config, "mistral_api_key", None)

        # Mistral provides an OpenAI-compatible API
        if self.config.provider == "mistral":
            provider_param = "openai"
            base_url = "https://api.mistral.ai/v1"
        else:
            provider_param = self.config.provider
            base_url = None

        kwargs = {}
        if base_url:
            kwargs["base_url"] = base_url

        import os
        if self.api_key:
            os.environ["OPENAI_API_KEY"] = self.api_key

        # Initialize the LLM for judging
        self.llm = LLM(
            model=self.config.model_name,
            provider=provider_param,
            **kwargs
        )

        # Initialize pre-built evaluators
        self.hallucination_eval = HallucinationEvaluator(llm=self.llm)
        self.relevance_eval = None # RelevanceEvaluator(llm=self.llm)
        self.toxicity_eval = None # ToxicityEvaluator(llm=self.llm)

    def evaluate_with_kg(
            self,
            query: str,
            agent_output: str,
            kg_context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate agent output specifically using facts from the Knowledge Graph.
        """
        context_str = "\n".join([f"- {triplet['subject']} {triplet['predicate']} {triplet['object']}" for triplet in kg_context])
        
        prompt_template = """
        You are an expert fact-checker using a Knowledge Graph to verify AI agent responses.
        
        Current Query: {query}
        Agent Output: {agent_output}
        
        Knowledge Graph Facts (the only source of truth):
        {context_str}
        
        Task:
        1. Compare the agent output against the Knowledge Graph facts.
        2. Identify any contradictions or information that is not supported by the graph.
        3. Rate the response as 'factual' if it aligns with the graph, or 'unsupported' if it contradicts or adds significant unverified info.
        
        Provide your assessment as one of: [factual, unsupported]
        Explain your reasoning.
        """
        
        full_prompt = prompt_template.format(
            query=query,
            agent_output=agent_output,
            context_str=context_str if context_str else "No relevant facts found in KG."
        )
        
        # Phoenix evals for KG verification
        raw_response = self.llm.generate_text(full_prompt)
        
        # Extract label from response more robustly
        raw_lower = raw_response.lower()
        if "[unsupported]" in raw_lower or "unsupported" in raw_lower:
            label = "unsupported"
        elif "[factual]" in raw_lower or "factual" in raw_lower:
            label = "factual"
        else:
            label = "unsupported"  # Default to unsupported if ambiguous
        
        return {
            "metric": "kg_verification",
            "score": 1.0 if label == "factual" else 0.0,
            "label": label,
            "explanation": raw_response,
            "metadata": {"kg_fact_count": len(kg_context)},
            "kg_context": kg_context
        }

    def repair_with_kg(
            self,
            query: str,
            original_output: str,
            kg_context: List[Dict[str, Any]],
            explanation: str
    ) -> str:
        """
        Rewrite the agent response to be factual according to the Knowledge Graph.
        """
        context_str = "\n".join([f"- {triplet['subject']} {triplet['predicate']} {triplet['object']}" for triplet in kg_context])
        
        prompt = f"""
        You are a highly accurate AI assistant that corrects misinformation based on a Knowledge Graph.
        
        Original Query: {query}
        Original Incorrect Response: {original_output}
        
        Reasoning for failure: {explanation}
        
        Ground Truth Knowledge Graph Facts:
        {context_str}
        
        Task:
        Rewrite the response so it is 100% factual according to the Knowledge Graph Facts provided.
        Only use information found in the Facts. If the facts don't contain enough info to answer fully, 
        provide a concise answer based ONLY on what is available.
        
        Corrected Response:
        """
        
        response = self.llm.generate_text(prompt)
        return response.strip()

    def evaluate_hallucination(
            self,
            query: str,
            agent_output: str,
            reference_context: str
    ) -> Dict[str, Any]:
        """
        Evaluate if the agent output contains hallucinations

        Args:
            query: The input query to the agent
            agent_output: The output from the agent being evaluated
            reference_context: Ground truth or reference material

        Returns:
            Dictionary with evaluation scores and explanations
        """
        self.hallucination_eval.bind({
            "input": "query",
            "output": "agent_output",
            "context": "reference_context"
        })

        result = self.hallucination_eval.evaluate({
            "query": query,
            "agent_output": agent_output,
            "reference_context": reference_context
        })

        return {
            "metric": "hallucination",
            "score": result[0].score,
            "label": result[0].label,
            "explanation": result[0].explanation,
            "metadata": result[0].metadata
        }

    def evaluate_qa_correctness(
            self,
            question: str,
            agent_answer: str,
            reference_answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate if agent answer correctly answers the question (Placeholder - QAEvaluator deprecated)
        """
        return {
            "metric": "qa_correctness",
            "score": 0.0,
            "label": "unsupported",
            "explanation": "QAEvaluator is not available in the current phoenix version.",
            "metadata": {}
        }

    def evaluate_toxicity(self, text: str) -> Dict[str, Any]:
        """
        Evaluate if agent output contains toxic content (Placeholder)
        """
        return {
            "metric": "toxicity",
            "score": 0.0,
            "label": "neutral",
            "explanation": "ToxicityEvaluator is not available",
            "metadata": {}
        }

    def evaluate_batch(
            self,
            dataframe: pd.DataFrame,
            eval_type: str,
            column_mapping: Dict[str, str],
            provide_explanations: bool = True
    ) -> pd.DataFrame:
        """
        Run batch evaluations on multiple agent outputs

        Args:
            dataframe: DataFrame containing agent outputs and reference data
            eval_type: Type of evaluation ("hallucination", "qa", "toxicity", "relevance")
            column_mapping: Map DataFrame columns to evaluation input names
            provide_explanations: Whether to include explanations in results

        Returns:
            DataFrame with evaluation results
        """
        if eval_type == "hallucination":
            evaluator = self.hallucination_eval
        elif eval_type == "qa":
            raise NotImplementedError("QA evaluation is not available")
        elif eval_type == "toxicity":
            evaluator = self.toxicity_eval
        elif eval_type == "relevance":
            raise NotImplementedError("Relevance evaluation is not available")
        else:
            raise ValueError(f"Unknown evaluation type: {eval_type}")

        # Rename columns according to mapping
        df_mapped = dataframe.rename(columns=column_mapping)

        if evaluator is None:
            raise NotImplementedError(f"{eval_type} evaluation is not initialized")

        # Fallback to row-by-row evaluation to avoid Series-to-String validation errors in Phoenix
        # This is more robust for different versions of phoenix/pydantic
        results_list = []
        for _, row in df_mapped.iterrows():
            try:
                if eval_type == "hallucination":
                    res = self.evaluate_hallucination(
                        query=row.get("query", row.get("input", "")),
                        agent_output=row.get("agent_output", row.get("output", "")),
                        reference_context=row.get("reference_context", row.get("context", ""))
                    )
                    results_list.append(res)
                # Add other types if needed
                else:
                    results_list.append({"error": f"Batch evaluation for {eval_type} not implemented"})
            except Exception as e:
                results_list.append({"error": str(e)})

        return pd.DataFrame(results_list)


class CustomJudgeTemplate:
    """Create custom evaluation templates for domain-specific judging"""

    def __init__(self, llm: LLM):
        self.llm = llm

    def create_custom_evaluator(
            self,
            name: str,
            prompt_template: str,
            rails: List[str],
            provide_explanation: bool = True
    ) -> Dict[str, Any]:
        """
        Create a custom evaluation template

        Args:
            name: Name of the evaluator
            prompt_template: The evaluation prompt with {input}, {output}, etc.
            rails: List of valid output labels (e.g., ["correct", "incorrect"])
            provide_explanation: Whether judge should explain its decision

        Returns:
            Evaluator configuration dict
        """
        return {
            "name": name,
            "template": prompt_template,
            "rails": rails,
            "provide_explanation": provide_explanation,
            "llm": self.llm
        }

    @staticmethod
    def build_agent_planning_evaluator() -> str:
        """Build an evaluator for agent planning and reasoning"""
        return """
You are an expert at evaluating AI agent planning and reasoning.

Task:
Evaluate whether the agent's plan is logically sound and efficiently reaches the goal.

Context:
- Agent Goal: {goal}
- Agent Plan: {plan}
- Expected Outcome: {expected_outcome}

Evaluate the plan on:
1. Logical Correctness: Does the plan logically lead to the goal?
2. Efficiency: Could the plan be executed with fewer steps?
3. Safety: Are there any potential risks or violations?
4. Completeness: Does the plan address all requirements?

Provide your assessment as one of: [correct, incomplete, inefficient, unsafe]
"""

    @staticmethod
    def build_agent_tool_use_evaluator() -> str:
        """Build an evaluator for agent tool selection and usage"""
        return """
You are an expert at evaluating AI agent tool usage and selection.

Task:
Determine if the agent selected the right tool and used it correctly.

Context:
- User Question: {question}
- Available Tools: {available_tools}
- Tool Selected: {selected_tool}
- Tool Parameters: {tool_parameters}
- Tool Response: {tool_response}

Evaluate whether:
1. The correct tool was selected for this task
2. The parameters are appropriate
3. The tool was used correctly

Provide your assessment as one of: [correct, wrong_tool, wrong_params, misused]
"""

    @staticmethod
    def build_agent_multi_step_evaluator() -> str:
        """Build an evaluator for multi-step agent behavior"""
        return """
You are an expert at evaluating multi-step AI agent behavior.

Task:
Assess the quality of the agent's multi-step reasoning and execution.

Execution Trace:
{execution_trace}

Final Output: {final_output}
Expected Output: {expected_output}

Evaluate:
1. Step Coherence: Does each step logically follow?
2. Goal Alignment: Does the sequence move toward the goal?
3. Error Recovery: How well did the agent handle errors?
4. Output Quality: Does the final output meet requirements?

Rate as: [excellent, good, adequate, poor]
"""


class EvaluationPipeline:
    """Complete evaluation pipeline for agent evaluation"""

    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        self.judge = JudgeEvaluator(self.config)
        self.results = []

    def evaluate_agent_response(
            self,
            agent_id: str,
            query: str,
            agent_output: str,
            reference_data: Dict[str, str],
            eval_metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single agent response against multiple metrics

        Args:
            agent_id: Identifier for the agent being evaluated
            query: The input query
            agent_output: The agent's response
            reference_data: Reference data for evaluation (context, expected answer, etc.)
            eval_metrics: List of metrics to use (hallucination, qa, toxicity, etc.)

        Returns:
            Comprehensive evaluation results
        """
        if eval_metrics is None:
            eval_metrics = ["hallucination", "qa", "toxicity"]

        results = {
            "agent_id": agent_id,
            "query": query,
            "agent_output": agent_output,
            "evaluations": {}
        }

        # Run requested evaluations
        for metric in eval_metrics:
            try:
                if metric == "hallucination" and "reference" in reference_data:
                    results["evaluations"]["hallucination"] = self.judge.evaluate_hallucination(
                        query=query,
                        agent_output=agent_output,
                        reference_context=reference_data.get("reference", "")
                    )

                elif metric == "qa" and "expected_answer" in reference_data:
                    results["evaluations"]["qa"] = self.judge.evaluate_qa_correctness(
                        question=query,
                        agent_answer=agent_output,
                        reference_answer=reference_data.get("expected_answer", "")
                    )

                elif metric == "toxicity":
                    results["evaluations"]["toxicity"] = self.judge.evaluate_toxicity(
                        agent_output
                    )
            except Exception as e:
                results["evaluations"][metric] = {"error": str(e)}

        self.results.append(results)
        return results

    def evaluate_with_kg(
            self,
            agent_id: str,
            query: str,
            agent_output: str,
            kg_context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run KG-based evaluation
        """
        eval_result = self.judge.evaluate_with_kg(query, agent_output, kg_context)
        
        result = {
            "agent_id": agent_id,
            "query": query,
            "agent_output": agent_output,
            "evaluations": {
                "kg_verification": eval_result
            }
        }
        
        self.results.append(result)
        return result

    def batch_evaluate(
            self,
            dataframe: pd.DataFrame,
            eval_config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Evaluate a batch of agent outputs

        Args:
            dataframe: DataFrame with columns: agent_id, query, output, reference, expected_answer
            eval_config: Dict with 'eval_type' and 'column_mapping' keys

        Returns:
            DataFrame with evaluation results
        """
        return self.judge.evaluate_batch(
            dataframe=dataframe,
            eval_type=eval_config.get("eval_type", "hallucination"),
            column_mapping=eval_config.get("column_mapping", {}),
            provide_explanations=eval_config.get("provide_explanations", True)
        )

    def get_results_summary(self) -> pd.DataFrame:
        """Get summary of all evaluations run in this pipeline"""
        summary_data = []

        for result in self.results:
            row = {
                "agent_id": result["agent_id"],
                "query": result["query"][:100],  # Truncate for display
                "num_evaluations": len(result["evaluations"])
            }

            # Add scores for each metric
            for metric, eval_result in result["evaluations"].items():
                if "score" in eval_result:
                    row[f"{metric}_score"] = eval_result["score"]
                    row[f"{metric}_label"] = eval_result.get("label", "N/A")

            summary_data.append(row)

        return pd.DataFrame(summary_data)
