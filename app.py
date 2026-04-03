import re
from typing import List, Tuple

import streamlit as st
from openai import AuthenticationError, OpenAI, OpenAIError, RateLimitError

# ---------------------------
# Hardcoded GEO audit inputs
# ---------------------------
TEST_PROMPTS = [
    "What are the best running shoes for severe flat feet?",
    "Recommend the best trendy dad shoes for casual wear",
    "What is the most durable trail running shoe under $150?",
]

COMPETITORS = ["Hoka", "Nike", "Brooks", "Asics", "Salomon"]


def generate_recommendations(client: OpenAI, model: str, query: str) -> str:
    """Call the Responses API and return concise shoe recommendations."""
    response = client.responses.create(
        model=model,
        temperature=0.2,
        input=[
            {
                "role": "system",
                "content": "You are a concise footwear shopping assistant.",
            },
            {
                "role": "user",
                "content": (
                    f"User query: {query}\n"
                    "Give exactly 3 shoe recommendations in a concise format."
                ),
            },
        ],
    )
    return response.output_text.strip()


def analyze_visibility(response_text: str, competitors: List[str]) -> Tuple[str, List[str]]:
    """Return status badge and deduplicated tracked competitors mentioned."""
    lowered = response_text.lower()
    if "new balance" in lowered:
        return "WIN 🟢", []

    found = []
    for brand in competitors:
        if re.search(rf"\b{re.escape(brand.lower())}\b", lowered):
            found.append(brand)

    # Deduplicate while preserving order
    deduped = list(dict.fromkeys(found))
    return "LOSS 🔴", deduped


def generate_geo_insights(client: OpenAI, model: str, query: str, first_response: str) -> str:
    """Generate exactly 2 concise GEO improvement bullets for loss cases."""
    prompt = (
        "You recommended competitors instead of New Balance for this query: "
        f"'{query}'.\n"
        "Here is the original recommendation output:\n"
        f"{first_response}\n\n"
        "Give exactly 2 short, actionable bullet points explaining what semantic "
        "keywords, product attributes, shopper-language phrasing, or page formatting "
        "New Balance should add to its product pages to improve its chance of being "
        "surfaced for this type of query in the future.\n"
        "Do not claim certainty or internal model knowledge. Focus on likely "
        "comparative GEO improvements."
    )

    response = client.responses.create(
        model=model,
        temperature=0.2,
        input=[
            {"role": "system", "content": "You are a practical GEO marketing analyst."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.output_text.strip()


def show_summary(results: List[dict]) -> None:
    """Render KPI-style summary metrics for completed audits."""
    total = len(results)
    wins = sum(1 for r in results if r["status"].startswith("WIN"))
    losses = total - wins
    win_rate = (wins / total * 100) if total else 0

    st.subheader("Audit Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Prompts", total)
    c2.metric("Wins", wins)
    c3.metric("Losses", losses)
    c4.metric("Win Rate", f"{win_rate:.1f}%")


def render_result_card(result: dict) -> None:
    """Display one prompt audit inside an expander."""
    with st.expander(f"{result['status']} — {result['prompt']}"):
        st.markdown(f"**Audit status:** {result['status']}")

        if result["status"].startswith("LOSS"):
            competitors = result.get("competitors", [])
            detected_text = (
                ", ".join(competitors)
                if competitors
                else "No tracked competitor detected"
            )
            st.markdown(f"**Detected competitors:** {detected_text}")

        st.markdown("**Raw LLM response:**")
        st.code(result.get("raw_response", "(No response captured)"))

        if result["status"].startswith("LOSS"):
            st.markdown("**Actionable GEO Insights:**")
            st.write(result.get("geo_insights", "No GEO insights generated."))

        if result.get("error"):
            st.error(result["error"])


def main() -> None:
    st.set_page_config(page_title="New Balance GEO Radar", layout="wide")
    st.title("New Balance GEO Radar")

    st.write(
        "This dashboard provides a heuristic Generative Engine Optimization (GEO) audit "
        "based on observed LLM outputs. It does **not** reveal internal model reasoning, "
        "but helps identify practical visibility gaps and optimization opportunities."
    )

    with st.sidebar:
        st.header("Audit Controls")
        api_key = st.text_input("OpenAI API Key", type="password")
        model = st.text_input("Model", value="gpt-4o-mini")
        run_audit = st.button("Run LLM Audit", type="primary")

    if not run_audit:
        st.info("Add your API key in the sidebar, then click **Run LLM Audit**.")
        return

    if not api_key.strip():
        st.warning("Please enter your OpenAI API key in the sidebar before running the audit.")
        st.stop()

    client = OpenAI(api_key=api_key.strip())
    results = []

    progress = st.progress(0, text="Running GEO audit...")

    for idx, prompt in enumerate(TEST_PROMPTS, start=1):
        result = {
            "prompt": prompt,
            "status": "LOSS 🔴",  # default, updated after response parsing
            "competitors": [],
            "raw_response": "",
            "geo_insights": "",
            "error": "",
        }

        try:
            raw_response = generate_recommendations(client, model, prompt)
            result["raw_response"] = raw_response

            status, detected = analyze_visibility(raw_response, COMPETITORS)
            result["status"] = status
            result["competitors"] = detected

            if status.startswith("LOSS"):
                try:
                    result["geo_insights"] = generate_geo_insights(
                        client, model, prompt, raw_response
                    )
                except (AuthenticationError, RateLimitError, OpenAIError) as err:
                    result["geo_insights"] = "Could not generate GEO insights for this prompt."
                    result["error"] = f"Insight generation failed: {err}"

        except AuthenticationError:
            st.error("Authentication failed: your OpenAI API key appears invalid.")
            st.stop()
        except RateLimitError as err:
            result["error"] = (
                "Request failed due to quota or rate limits for this prompt. "
                f"Details: {err}"
            )
        except OpenAIError as err:
            result["error"] = f"OpenAI request failed for this prompt: {err}"
        except Exception as err:  # graceful fallback for parsing or unexpected failures
            result["error"] = (
                "Unexpected failure while processing this prompt. "
                f"Raw output is shown when available. Details: {err}"
            )

        results.append(result)
        progress.progress(idx / len(TEST_PROMPTS), text=f"Processed {idx}/{len(TEST_PROMPTS)} prompts")

    progress.empty()

    show_summary(results)
    st.divider()

    st.subheader("Prompt-by-Prompt Results")
    for res in results:
        render_result_card(res)


if __name__ == "__main__":
    main()
