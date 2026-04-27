from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from crewai import Agent, Task, Crew, LLM
from crewai.process import Process
from crewai_tools import SerperDevTool
import os
import json
import time
import fcntl
from datetime import datetime

# ── Load .env only in local development ──────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__)
CORS(app)

# ══════════════════════════════════════════════════════════════════════════════
#  API KEYS
# ══════════════════════════════════════════════════════════════════════════════
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

if not GEMINI_API_KEY or not SERPER_API_KEY:
    raise EnvironmentError(
        "Missing required environment variables. "
        "Set GEMINI_API_KEY and SERPER_API_KEY in the environment or .env."
    )

os.environ["SERPER_API_KEY"] = SERPER_API_KEY

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL INITIALISATION
#
#  lite_llm → gemini-2.0-flash-lite   fast · high RPM · low complexity
#  pro_llm  → gemini-2.5-flash        quality · reserved for final work
#  search_llm → gemini-2.0-flash-lite  used for standalone search agent
#               (gemma-2-9b-it removed — not reliably available via same key)
# ══════════════════════════════════════════════════════════════════════════════

lite_llm = LLM(
    model="gemini/gemini-2.5-flash-lite",
    temperature=0.2,
    api_key=GEMINI_API_KEY,
)

pro_llm = LLM(
    model="gemini/gemini-2.5-flash",
    temperature=0.3,
    api_key=GEMINI_API_KEY,
)

# Search agent uses lite to preserve pro quota
search_llm = LLM(
    model="gemini/gemini-2.5-flash-lite",
    temperature=0.1,
    api_key=GEMINI_API_KEY,
)

# ── Tools ─────────────────────────────────────────────────────────────────────
search_tool = SerperDevTool()

# ══════════════════════════════════════════════════════════════════════════════
#  MEMORY CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_memory_path() -> str:
    env_path = os.getenv("MEMORY_PATH")
    if env_path:
        print(f"[Memory] Using env-specified path: {env_path}")
        return env_path

    tmp_path = "/tmp/memory.json"
    try:
        with open(tmp_path, "a", encoding="utf-8") as f:
            pass
        print(f"[Memory] Using /tmp path: {tmp_path}")
        return tmp_path
    except IOError:
        pass

    base_dir   = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(base_dir, "memory.json")
    print(f"[Memory] Using local path: {local_path}")
    return local_path


MEMORY_FILE        = _resolve_memory_path()
MAX_MEMORY_ENTRIES = 15

if not os.path.exists(MEMORY_FILE):
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
        print(f"[Memory] Created empty memory file at: {MEMORY_FILE}")
    except IOError as e:
        print(f"[Memory] WARNING: Could not pre-create memory file: {e}")


# ═════════════════════════════════════════════════════════════════════════════
#  MEMORY HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def load_memory() -> list:
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            raw = f.read().strip()
            if not raw:
                return []
            data = json.loads(raw)
            entries = data if isinstance(data, list) else []
            print(f"[Memory] Loaded {len(entries)} entries from {MEMORY_FILE}")
            return entries
    except json.JSONDecodeError as exc:
        print(f"[Memory] JSON corrupt — resetting. Reason: {exc}")
        return []
    except IOError as exc:
        print(f"[Memory] Cannot read {MEMORY_FILE}: {exc}")
        return []


def save_memory(entry: dict) -> bool:
    print(f"[Memory] Saving entry for company: {entry.get('company', 'unknown')}")

    if not isinstance(entry, dict) or not entry:
        print("[Memory] ERROR: entry is empty or not a dict — skipping.")
        return False

    try:
        parent = os.path.dirname(MEMORY_FILE)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        memory = load_memory()
        memory.append(entry)

        if len(memory) > MAX_MEMORY_ENTRIES:
            memory = memory[-MAX_MEMORY_ENTRIES:]

        tmp_file = MEMORY_FILE + ".tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX)
            except Exception:
                pass

            json.dump(memory, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

            try:
                fcntl.flock(f, fcntl.LOCK_UN)
            except Exception:
                pass

        os.replace(tmp_file, MEMORY_FILE)

        verify = load_memory()
        if len(verify) > 0:
            print(f"[Memory] ✅ Saved. Total entries now: {len(verify)}")
            return True
        else:
            print("[Memory] ❌ Write seemed OK but file is empty on verify!")
            return False

    except PermissionError as exc:
        print(f"[Memory] ❌ PERMISSION DENIED: {MEMORY_FILE} — {exc}")
        return False
    except IOError as exc:
        print(f"[Memory] ❌ IOError writing to {MEMORY_FILE}: {exc}")
        return False
    except Exception as exc:
        print(f"[Memory] ❌ Unexpected: {type(exc).__name__}: {exc}")
        return False


def get_recent_memory(n: int = 3) -> list:
    entries = load_memory()
    recent  = entries[-n:]
    print(f"[Memory] Returning {len(recent)} recent entries.")
    return recent


def format_memory_context(entries: list) -> str:
    if not entries:
        return "No prior meeting history available."
    parts = []
    for i, e in enumerate(entries, 1):
        ts        = e.get("timestamp", "unknown time")
        company   = e.get("company",   "N/A")
        objective = e.get("objective", "N/A")
        summary   = e.get("summary",   "N/A")
        parts.append(
            f"[Past Meeting {i} | {ts}]\n"
            f"  Company   : {company}\n"
            f"  Objective : {objective}\n"
            f"  Takeaway  : {summary}"
        )
    return "\n\n".join(parts)


# ═════════════════════════════════════════════════════════════════════════════
#  DECISION PARSER
# ═════════════════════════════════════════════════════════════════════════════

def parse_decision(text: str) -> dict:
    upper = text.upper()

    if "SEARCH: ALWAYS" in upper:
        search_mode = "ALWAYS"
    elif "SEARCH: LIGHT" in upper:
        search_mode = "LIGHT"
    else:
        search_mode = "MINIMAL"
    use_search = search_mode != "MINIMAL"

    use_memory = "MEMORY: NO" not in upper

    if "PRIORITY: INDUSTRY" in upper:
        priority = "Industry"
    elif "PRIORITY: STRATEGY" in upper:
        priority = "Strategy"
    else:
        priority = "Context"

    if "DEPTH: DEEP" in upper:
        depth = "DEEP"
    elif "DEPTH: SHORT" in upper:
        depth = "SHORT"
    else:
        depth = "NORMAL"

    return {
        "use_search":  use_search,
        "search_mode": search_mode,
        "use_memory":  use_memory,
        "priority":    priority,
        "depth":       depth,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  SMART MODEL ROUTER  —  rate-limit detection + fallback
# ═════════════════════════════════════════════════════════════════════════════

_RATE_LIMIT_KEYWORDS = [
    "429", "rate_limit", "Rate limit", "quota", "Quota",
    "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "overloaded",
    "too many requests", "exceeded",
]

def _is_rate_error(msg: str) -> bool:
    return any(k.lower() in msg.lower() for k in _RATE_LIMIT_KEYWORDS)


def kickoff_with_retry(crew, retries: int = 3, base_wait: int = 12):
    for attempt in range(retries):
        try:
            return crew.kickoff()
        except Exception as e:
            msg = str(e)
            if _is_rate_error(msg) and attempt < retries - 1:
                wait = base_wait * (2 ** attempt)
                print(f"[Retry] Rate limit hit — waiting {wait}s "
                      f"(attempt {attempt + 1}/{retries})")
                time.sleep(wait)
                continue
            raise


def kickoff_with_model_fallback(crew_builder_fn, high_quality: bool = False):
    primary   = pro_llm  if high_quality else lite_llm
    secondary = lite_llm if high_quality else None

    crew = crew_builder_fn(primary)
    try:
        return kickoff_with_retry(crew)
    except Exception as e:
        if high_quality and secondary and _is_rate_error(str(e)):
            print("[ModelRouter] gemini-2.5-flash rate-limited → "
                  "falling back to gemini-2.0-flash-lite")
            time.sleep(8)
            crew = crew_builder_fn(secondary)
            return kickoff_with_retry(crew)
        raise


# ═════════════════════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "models": {
            "lite":   "gemini/gemini-2.5-flash-lite",
            "pro":    "gemini/gemini-2.5-flash",
            "search": "gemini/gemini-2.0-flash-lite",
        },
        "timestamp": datetime.utcnow().isoformat(),
    }), 200


@app.route("/debug-memory")
def debug_memory():
    test_entry = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "company":   "DEBUG_TEST",
        "objective": "Verify memory write works",
        "summary":   "Test entry to confirm save_memory() is functional.",
    }
    saved  = save_memory(test_entry)
    loaded = load_memory()
    return jsonify({
        "memory_file_path": MEMORY_FILE,
        "file_exists":      os.path.exists(MEMORY_FILE),
        "is_writable":      os.access(os.path.dirname(MEMORY_FILE) or ".", os.W_OK),
        "save_returned":    saved,
        "total_entries":    len(loaded),
        "entries":          loaded,
    })


@app.route("/run-agent", methods=["POST"])
def run_agent():
    data = request.get_json(force=True)

    required_fields = [
        "company_name", "meeting_objective",
        "attendees", "meeting_duration", "focus_areas",
    ]
    missing = [f for f in required_fields if not data.get(f)]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    company_name      = data["company_name"]
    meeting_objective = data["meeting_objective"]
    attendees         = data["attendees"]
    meeting_duration  = int(data["meeting_duration"])
    focus_areas       = data["focus_areas"]

    start_time = time.time()

    try:
        # ══════════════════════════════════════════════════════════════════════
        #  PHASE 0 — Load Memory
        # ══════════════════════════════════════════════════════════════════════
        recent_memory  = get_recent_memory(3)
        memory_context = format_memory_context(recent_memory)
        has_memory     = len(recent_memory) > 0

        # ══════════════════════════════════════════════════════════════════════
        #  PHASE 1 — Decision Agent  (lite_llm)
        # ══════════════════════════════════════════════════════════════════════
        def build_decision_crew(llm):
            decision_agent = Agent(
                role="Meeting Prep Orchestrator",
                goal=(
                    "Decide how the downstream agents should use web search, "
                    "memory, and which analysis area to prioritize for this meeting."
                ),
                backstory=(
                    "You coordinate a pipeline of expert agents. You never guess. "
                    "You output only a strict decision block that configures search, "
                    "memory, and analysis depth."
                ),
                verbose=False,
                allow_delegation=False,
                llm=llm,
            )

            decision_task = Task(
                description=f"""
                You are orchestrating a multi-agent system that prepares executive meeting briefs.
                MEETING REQUEST:
                - Company       : {company_name}
                - Objective     : {meeting_objective}
                - Attendees     : {attendees}
                - Duration      : {meeting_duration} minutes
                - Focus areas   : {focus_areas}
                PAST MEETING MEMORY:
                {memory_context}
                You must output EXACTLY these 5 lines (no extra text):
                SEARCH: ALWAYS or LIGHT or MINIMAL
                MEMORY: YES or NO
                PRIORITY: Context or Industry or Strategy
                DEPTH: SHORT or NORMAL or DEEP
                REASONING: one concise sentence explaining all decisions
                DECISION RULES:
                - SEARCH:
                  - ALWAYS  → new or complex company/industry; need 4–5 web searches
                  - LIGHT   → known company; 2–3 web searches are enough
                  - MINIMAL → mostly internal topic; 1 search just to validate facts
                - MEMORY:
                  - YES if past memory mentions this company or very similar ones
                  - NO if memory is empty or clearly unrelated
                - PRIORITY:
                  - Context   → understanding the company is the biggest gap
                  - Industry  → market / competition / trends are most important
                  - Strategy  → agenda and talking points need most work
                - DEPTH:
                  - SHORT   → meeting_duration ≤ 30 minutes
                  - NORMAL  → 31–60 minutes
                  - DEEP    → >60 minutes; brief should be very detailed
                """,
                agent=decision_agent,
                expected_output="Exactly 5 lines: SEARCH, MEMORY, PRIORITY, DEPTH, REASONING.",
            )

            return Crew(
                agents=[decision_agent],
                tasks=[decision_task],
                verbose=False,
                max_rpm=9,
                max_execution_time=45,
                process=Process.sequential,
            )

        decision_result = kickoff_with_model_fallback(build_decision_crew, high_quality=False)
        decision_text   = str(decision_result).strip()
        decision_flags  = parse_decision(decision_text)

        use_search  = decision_flags["use_search"]
        search_mode = decision_flags["search_mode"]
        use_memory  = decision_flags["use_memory"] and has_memory
        priority    = decision_flags["priority"]
        depth       = decision_flags["depth"]

        # ── Memory injection ──────────────────────────────────────────────────
        memory_injection = (
            f"""
PRIOR MEETING CONTEXT (MANDATORY TO CONSIDER):
{memory_context}
You MUST:
- Reuse relevant insights from past meetings where helpful.
- Maintain continuity in recommendations.
- Avoid contradicting past decisions unless clearly justified.
"""
            if use_memory else ""
        )

        # ── Tool assignments per agent ────────────────────────────────────────
        context_tools  = [search_tool] if use_search else []
        industry_tools = [search_tool] if (use_search and priority == "Industry") else []

        # ── Search instructions based on mode ────────────────────────────────
        if search_mode == "ALWAYS":
            context_search_instruction = (
                "MANDATORY: Use the search tool 3–5 times for company profile, "
                "recent news, products, and competitors."
            )
            industry_search_instruction = (
                "MANDATORY: Use the search tool 2–3 times for industry trends, "
                "market size, and key competitors."
            )
        elif search_mode == "LIGHT":
            context_search_instruction = (
                "MANDATORY: Use the search tool at least 2 times for company "
                "overview and latest news."
            )
            industry_search_instruction = (
                "OPTIONAL: Use the search tool at most 1–2 times if needed."
            )
        else:  # MINIMAL
            context_search_instruction = (
                "MANDATORY: Use the search tool exactly once to validate basic facts."
            )
            industry_search_instruction = (
                "Do NOT perform additional web searches; rely on provided context."
            )

        # ══════════════════════════════════════════════════════════════════════
        #  PHASE 2 — Main 4-Agent Crew
        # ══════════════════════════════════════════════════════════════════════

        def build_main_crew(context_llm, industry_llm, strategy_llm, brief_llm):

            context_analyzer = Agent(
                role="Meeting Context Specialist",
                goal="Produce a concise, factual company + meeting context summary.",
                backstory=(
                    "You quickly understand complex business contexts and identify "
                    "only the most critical, verifiable information. You prefer "
                    "specific numbers, dates, and named entities over vague statements."
                ),
                verbose=False,
                allow_delegation=False,
                llm=context_llm,
                tools=context_tools,
            )

            industry_insights_generator = Agent(
                role="Industry Expert",
                goal="Provide a short but insightful industry overview and key trends.",
                backstory=(
                    "You are a seasoned industry analyst who spots important trends, "
                    "competitors, opportunities, and risks. You ground your analysis "
                    "in concrete facts and examples."
                ),
                verbose=False,
                allow_delegation=False,
                llm=industry_llm,
                tools=industry_tools,
            )

            strategy_formulator = Agent(
                role="Meeting Strategist",
                goal="Design a tight, outcome-focused meeting strategy and agenda.",
                backstory=(
                    "You create practical, time-boxed agendas that align with "
                    "strategic goals and stakeholder interests. Every recommendation "
                    "is specific, actionable, and time-bound."
                ),
                verbose=False,
                allow_delegation=False,
                llm=strategy_llm,
            )

            executive_briefing_creator = Agent(
                role="Communication Specialist",
                goal="Synthesize everything into a clear, actionable executive brief.",
                backstory=(
                    "You distill complex analysis into crisp, high-impact talking points, "
                    "Q&A prep, and recommendations for C-level executives."
                ),
                verbose=False,
                allow_delegation=False,
                llm=brief_llm,
            )

            context_analysis_task = Task(
                description=f"""
                You are preparing for a meeting with {company_name}.
                Search instruction: {context_search_instruction}
                You MUST follow this search instruction exactly if the search tool is available.
                Produce a context summary that covers:
                - Company snapshot: what they do, scale, geography.
                - 1–3 recent notable news items or strategic moves.
                - Key products / services relevant to this meeting.
                - 3–5 major direct competitors.
                Meeting details:
                - Objective  : {meeting_objective}
                - Attendees  : {attendees}
                - Duration   : {meeting_duration} minutes
                - Focus areas: {focus_areas}
                { "PRIORITY FLAG: Context analysis is the highest priority—go deeper here." if priority == "Context" else "" }
                {memory_injection}
                Requirements:
                - Include specific numbers (revenue, employees, etc.) where possible.
                - Include dates for major events or news.
                - Avoid generic phrases like "leverage synergies" or "move the needle".
                - Target length: 400–700 words.
                Output style: markdown, with clear headings and bullet points.
                """,
                agent=context_analyzer,
                expected_output="A concise markdown summary of company + meeting context.",
            )

            industry_analysis_task = Task(
                description=f"""
                Based on the previous context analysis for {company_name} and the
                meeting objective: "{meeting_objective}", provide an industry-level view.
                Search instruction: {industry_search_instruction}
                Focus on:
                - 3–5 key industry or market trends relevant to this meeting.
                - Competitive landscape and where {company_name} roughly fits.
                - 3–5 main opportunities {company_name} could pursue.
                - 3–5 main risks or threats they should watch.
                { "PRIORITY FLAG: Industry analysis is the highest priority—go deeper here." if priority == "Industry" else "" }
                {memory_injection}
                Requirements:
                - Use examples of competitors and adjacent players.
                - Include any relevant regulations or technology trends.
                - Target length: 400–700 words.
                Output: markdown with clear headings and bullet points.
                """,
                agent=industry_insights_generator,
                expected_output="A short, insightful markdown industry analysis.",
            )

            strategy_development_task = Task(
                description=f"""
                Using the prior analyses (context + industry), design a concrete strategy
                for the {meeting_duration}-minute meeting with {company_name}.
                Do NOT perform web searches.
                Produce:
                1. A time-boxed agenda (section name + minutes) that sums to {meeting_duration} minutes.
                2. 3–7 key talking points the host should definitely cover.
                3. For each focus area in: "{focus_areas}", propose 1–3 concrete strategies.
                { "PRIORITY FLAG: Strategy development is the highest priority—be especially detailed and actionable." if priority == "Strategy" else "" }
                {memory_injection}
                Requirements:
                - Every agenda item must have a clear outcome or purpose.
                - Every recommendation must specify WHO does WHAT by WHEN.
                - Avoid vague language such as "discuss opportunities" or "align on strategy".
                - Target length: 600–800 words.
                Output: markdown, bullet-point heavy.
                """,
                agent=strategy_formulator,
                expected_output=(
                    "A succinct markdown meeting strategy with time-boxed agenda and talking points."
                ),
            )

            executive_brief_task = Task(
                description=f"""
                Synthesize EVERYTHING into a single executive brief for the meeting
                with {company_name}.
                You will receive prior analyses (context, industry, strategy). Use them all.
                IMPORTANT: Output ONLY the final brief in markdown.
                No internal reasoning, planning text, or preamble.
                Required structure:
                # Executive Summary
                - 3–6 bullet points capturing the meeting objective and context.
                ## Company & Industry Snapshot
                - Short bullets on who {company_name} is and key market dynamics.
                ## Meeting Goals & Success Criteria
                - 3–5 clearly stated, measurable goals.
                - How the host will know the meeting succeeded.
                ## Recommended Agenda & Key Talking Points
                - Tight recap of the time-boxed agenda (must total {meeting_duration} minutes).
                - Bullet list of the most important talking points, tied to data or examples.
                ## Anticipated Questions & Prepared Answers
                - 5–10 likely questions from attendees based on their roles.
                - 1–3 sentence answer for each, grounded in prior analysis.
                ## Strategic Recommendations & Next Steps
                - 3–5 actionable post-meeting recommendations.
                - Suggested next steps and rough timelines.
                Requirements:
                - Markdown headings + bullets.
                - Target length: 900–1300 words.
                - Use specific numbers, dates, and names where possible.
                - Maintain a professional, concise tone suitable for C-level executives.
                """,
                agent=executive_briefing_creator,
                expected_output=(
                    "A complete markdown executive brief ready to share before the meeting."
                ),
            )

            return Crew(
                agents=[
                    context_analyzer,
                    industry_insights_generator,
                    strategy_formulator,
                    executive_briefing_creator,
                ],
                tasks=[
                    context_analysis_task,
                    industry_analysis_task,
                    strategy_development_task,
                    executive_brief_task,
                ],
                verbose=False,
                max_rpm=4,
                max_execution_time=300,
                process=Process.sequential,
            )

        # ── Run main crew ─────────────────────────────────────────────────────
        try:
            main_crew = build_main_crew(
                context_llm  = lite_llm,
                industry_llm = lite_llm,
                strategy_llm = pro_llm,
                brief_llm    = pro_llm,
            )
            main_result = kickoff_with_retry(main_crew)

        except Exception as e:
            if _is_rate_error(str(e)):
                print("[ModelRouter] gemini-2.5-flash quota hit in main crew → "
                      "full gemini-2.0-flash-lite fallback")
                time.sleep(15)
                main_crew = build_main_crew(
                    context_llm  = lite_llm,
                    industry_llm = lite_llm,
                    strategy_llm = lite_llm,
                    brief_llm    = lite_llm,
                )
                main_result = kickoff_with_retry(main_crew)
            else:
                raise

        raw_brief = str(main_result).strip()

        # ══════════════════════════════════════════════════════════════════════
        #  PHASE 3 — Reflection Agent  (pro_llm → lite fallback)
        # ══════════════════════════════════════════════════════════════════════
        def build_reflection_crew(llm):
            reflection_agent = Agent(
                role="Executive Communications Editor",
                goal=(
                    "Polish the executive brief so it is maximally clear, "
                    "detailed, and actionable — ready to send to a C-suite audience."
                ),
                backstory=(
                    "You are a senior communications editor for high-stakes documents. "
                    "You remove redundancy, sharpen vague language, ensure every "
                    "recommendation is specific, and guarantee logical flow."
                ),
                verbose=False,
                allow_delegation=False,
                llm=llm,
            )

            reflection_task = Task(
                description=f"""
                Review and improve the following executive meeting brief.
                Editing checklist — apply EVERY item:
                1. Remove any duplicate or repeated information across sections.
                2. Sharpen vague language into specific, concrete statements.
                3. Make every recommendation actionable (who does what, by when).
                4. Fix any factual or logical inconsistencies between sections.
                5. Ensure the brief flows logically: Summary → Context → Goals →
                   Agenda → Q&A → Next Steps.
                6. Cut filler words and padding, but keep important detail.
                7. Keep all original markdown section headings intact.
                8. Do NOT add new sections — only improve existing content.
                9. Target length: 1100–1300 words (expand or compress as needed).
                ORIGINAL BRIEF TO IMPROVE:
                ---
                {raw_brief}
                ---
                OUTPUT: Return ONLY the improved markdown brief.
                No commentary, no preamble, no "Here is the improved version:" header.
                """,
                agent=reflection_agent,
                expected_output="An improved, polished executive brief in clean markdown format.",
            )

            return Crew(
                agents=[reflection_agent],
                tasks=[reflection_task],
                verbose=False,
                max_rpm=4,
                max_execution_time=120,
                process=Process.sequential,
            )

        reflection_result = kickoff_with_model_fallback(build_reflection_crew, high_quality=True)
        final_brief       = str(reflection_result).strip()

        # ══════════════════════════════════════════════════════════════════════
        #  PHASE 4 — Persist to Memory
        # ══════════════════════════════════════════════════════════════════════
        try:
            summary_preview = (
                final_brief[:350]
                .replace("\n", " ")
                .replace("#", "")
                .strip()
            )

            saved = save_memory({
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                "company":   company_name,
                "objective": meeting_objective,
                "summary":   summary_preview,
            })

            if not saved:
                print("[Memory] ⚠️ Entry was NOT saved — check logs above.")

        except Exception as mem_exc:
            print(f"[Memory] ❌ Phase 4 exception: {type(mem_exc).__name__}: {mem_exc}")

        elapsed = round(time.time() - start_time, 2)

        return jsonify({
            "result":      final_brief,
            "decision":    decision_text,
            "memory_used": use_memory,
            "flags":       decision_flags,
            "models": {
                "lite":   "gemini/gemini-2.0-flash-lite",
                "pro":    "gemini/gemini-2.5-flash",
                "search": "gemini/gemini-2.0-flash-lite",
            },
            "elapsed_sec": elapsed,
        }), 200

    except Exception as exc:
        msg = str(exc)

        if _is_rate_error(msg):
            return jsonify({
                "error": (
                    "⚠️ Gemini API rate limit reached. "
                    "Please wait 30–60 seconds and try again."
                )
            }), 429

        if "timed out" in msg.lower() or "timeout" in msg.lower():
            return jsonify({
                "error": (
                    "⏱️ The agent pipeline took too long to respond. "
                    "Try again — it usually succeeds on a second attempt."
                )
            }), 503

        return jsonify({"error": f"Agent error: {msg}"}), 500


# ═════════════════════════════════════════════════════════════════════════════
#  STANDALONE SEARCH AGENT  — now with retry + elapsed time
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/search-links", methods=["POST"])
def search_links():
    """
    Standalone link-search endpoint.
    Expects JSON: { "query": "..." }
    Returns: { query, links, model, elapsed_sec }
    """
    data = request.get_json(force=True)
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Missing or empty 'query' field."}), 400

    start_time = time.time()

    try:
        search_agent = Agent(
            role="Link Discovery Specialist",
            goal=f"Find the most relevant and high-quality links for: {query}",
            backstory=(
                "You are an expert at navigating the web. You provide only the most "
                "authoritative and useful links, avoiding spam or low-quality sources. "
                "For each result you return the page title, a one-sentence description, "
                "and the full URL."
            ),
            tools=[search_tool],
            llm=search_llm,          # uses lite_llm — preserves pro quota
            verbose=False,
            allow_delegation=False,
        )

        search_task = Task(
            description=(
                f"Search the web and find the top 5–8 most relevant, authoritative "
                f"links related to: '{query}'.\n\n"
                f"For each result provide:\n"
                f"- **Title**: the page title\n"
                f"- **URL**: the full link\n"
                f"- **Summary**: one sentence describing what the page covers\n\n"
                f"Format the output as a clean markdown list."
            ),
            expected_output=(
                "A markdown list of 5–8 links, each with title, URL, and one-sentence summary."
            ),
            agent=search_agent,
        )

        crew = Crew(
            agents=[search_agent],
            tasks=[search_task],
            verbose=False,
            max_rpm=9,
            max_execution_time=60,
            process=Process.sequential,
        )

        result  = kickoff_with_retry(crew, retries=2, base_wait=8)
        elapsed = round(time.time() - start_time, 2)

        return jsonify({
            "query":       query,
            "links":       str(result).strip(),
            "model":       "gemini/gemini-2.0-flash-lite",
            "elapsed_sec": elapsed,
        }), 200

    except Exception as e:
        msg = str(e)
        elapsed = round(time.time() - start_time, 2)

        if _is_rate_error(msg):
            return jsonify({
                "error":       "⚠️ Rate limit reached. Please wait 30 seconds and try again.",
                "elapsed_sec": elapsed,
            }), 429

        return jsonify({
            "error":       f"Search agent error: {msg}",
            "elapsed_sec": elapsed,
        }), 500


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=7860)