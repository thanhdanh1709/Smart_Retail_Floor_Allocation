# CROSS-AGENT CONTEXT HANDOFF PROTOCOL

## Scope

This protocol applies to BOTH:
- OpenAI Codex
- Claude Code

Both agents MUST maintain HANDOFF.md as the shared
execution context of this repository.

Existing ChatGPT Project and Playwright MCP rules
remain applicable to Codex.

Claude Code MUST NOT assume it has access to
Codex's ChatGPT Project browser session.

For major architectural decisions, Claude Code
must follow existing approved decisions or
escalate proposed changes for approval.

## 1. Mandatory Context Loading

BEFORE starting any implementation, BOTH agents MUST:

1. Read AGENTS.md.
2. Read HANDOFF.md completely.
3. Inspect git status.
4. Inspect recent commits.
5. Compare HANDOFF.md against actual source code.
6. Identify the active task and remaining work.
7. Identify unresolved blockers or conflicts.

Never start implementation using stale context.

If HANDOFF.md is missing, initialize it before
making significant implementation changes.

## 2. Mandatory Context Checkpoints

BOTH agents MUST update HANDOFF.md immediately when:

- A meaningful implementation step is completed.
- A technical or architectural decision is made.
- A bug or blocker is discovered or resolved.
- Task scope or implementation strategy changes.
- Relevant tests or builds are executed.
- A task is paused, interrupted, or transferred.
- An implementation task is completed.

Do not defer all updates until session termination.

Minor edits within the same implementation step
may be grouped into one checkpoint.

## 3. Mandatory Handoff Information

Every context update MUST accurately record:

- Current task
- Current agent: Codex or Claude Code
- Current branch
- Last verified commit
- Completed work
- Work in progress
- Modified files
- Technical decisions and reasoning
- Known issues and blockers
- Build and test commands with actual results
- Exact next action
- Relevant risks and uncertainties

Preserve important historical decisions.
Do not rewrite history to hide failures.

## 4. Bidirectional Transfer

### Codex to Claude Code

Codex MUST:
1. Complete the current safe checkpoint.
2. Update HANDOFF.md.
3. Record unfinished work and next actions.
4. Report uncommitted changes.
5. Confirm the handoff is ready.

Claude Code MUST:
1. Read the latest HANDOFF.md.
2. Inspect the actual repository state.
3. Verify previous work.
4. Continue unfinished tasks.
5. Maintain the same HANDOFF.md.

### Claude Code to Codex

Claude Code MUST follow the same handoff procedure.

Codex MUST:
1. Read the latest HANDOFF.md.
2. Verify the repository state.
3. Identify newly completed work.
4. Identify remaining tasks.
5. Continue using the existing Codex workflow.
6. Consult ChatGPT Project when required.

## 5. Context Consistency

Source code and verified build/test results
take precedence over outdated HANDOFF.md entries.

When inconsistencies are discovered:
1. Record the discrepancy.
2. Inspect the implementation.
3. Correct inaccurate context.
4. Preserve meaningful historical information.
5. Avoid duplicating completed work.

Never fabricate results or decisions.

## 6. Concurrent Development

- Never overwrite another developer's changes.
- Use separate branches for concurrent work.
- Never blindly overwrite HANDOFF.md.
- Reconcile conflicting handoff entries before merge.
- Preserve relevant context from both agents.

## 7. Mandatory Completion Gate

Before reporting implementation complete,
BOTH agents MUST:

1. Review all code changes.
2. Update HANDOFF.md.
3. Record test status.
4. Record remaining issues.
5. Record the next actionable step.
6. Verify the handoff reflects actual code.

If context is incomplete, report:

CONTEXT_NOT_SYNCHRONIZED

Do not claim the handoff is ready.

## 8. Session Interruption Recovery

After a session restart or context reset:

1. Re-read AGENTS.md.
2. Re-read HANDOFF.md.
3. Inspect git status and recent commits.
4. Verify relevant source code.
5. Resume from the last verified checkpoint.

Do not rely on previous chat memory alone.

## 9. Security

Never write secrets, credentials, tokens,
cookies, or private keys into HANDOFF.md.

Do not commit or push without authorization.

# PROJECT AI WORKFLOW

## Project binding

This repository is permanently associated with one ChatGPT Project.

CHATGPT_PROJECT_NAME:
Smart_Retail_Floor_Allocation

CHATGPT_PROJECT_URL:
https://chatgpt.com/g/g-p-6acb393fcd2481918628d3dd31a6592b/project

CHATGPT_BRIDGE_URL:
https://chatgpt.com/g/g-p-6acb393fcd2481918628d3dd31a6592b-smart-retail-floor-allocation/c/6acb3ac2-0994-83e9-8032-e304752d0264

## Core operating model

This project uses a two-agent workflow:

- ChatGPT Web inside the configured ChatGPT Project is the PRIMARY reasoning, research, architecture, planning, debugging, and review agent.
- Codex is the EXECUTION agent responsible mainly for opening ChatGPT Project, gathering repository context, editing code, running commands, building, testing, collecting results, and reporting those results back to ChatGPT Project.

The default control flow is:

USER REQUEST
→ CODEX RECEIVES TASK
→ CODEX OPENS CHATGPT PROJECT
→ CHATGPT ANALYZES / RESEARCHES / PLANS
→ CODEX IMPLEMENTS
→ CODEX BUILDS / TESTS
→ CODEX SENDS RESULTS TO CHATGPT
→ CHATGPT REVIEWS / DECIDES NEXT STEP
→ CODEX CONTINUES
→ FINAL CHATGPT REVIEW
→ COMPLETION

For non-trivial work, Codex must not skip ChatGPT Project consultation.


## Strict project isolation

All ChatGPT Web work for this repository MUST happen inside:

CHATGPT_PROJECT_NAME

Codex must NEVER intentionally use:

- a normal ChatGPT chat outside the configured Project
- Temporary Chat
- another ChatGPT Project
- an unrelated existing conversation
- a new chat created from the ChatGPT home page

If the configured ChatGPT Project cannot be verified, Codex must stop ChatGPT consultation instead of falling back to a normal chat.


## Roles

### ChatGPT Web — PRIMARY DECISION MAKER

ChatGPT Web inside the configured ChatGPT Project is the main technical decision maker for this repository.

ChatGPT Web is responsible for:

- understanding the user's request
- clarifying the real technical objective from available context
- analyzing repository architecture from context supplied by Codex
- performing technical research
- researching current documentation when needed
- comparing libraries, frameworks, APIs, protocols, and implementation approaches
- making architecture decisions
- deciding implementation strategy
- identifying affected files
- identifying files that should be created, modified, moved, or removed
- defining component responsibilities
- defining data flow
- defining state flow
- defining interfaces and APIs
- defining algorithms and data structures
- defining concurrency strategy
- defining memory and buffering strategy
- defining performance strategy
- defining error handling
- defining compatibility requirements
- defining hardware/software integration requirements
- designing UI/UX behavior when applicable
- analyzing compiler errors
- analyzing test failures
- analyzing runtime errors and logs
- performing root-cause analysis
- deciding corrective actions
- defining build steps
- defining test steps
- defining acceptance criteria
- reviewing Codex implementation results
- checking for regressions
- reviewing performance
- reviewing memory usage
- reviewing concurrency
- reviewing security
- reviewing maintainability
- reviewing edge cases
- deciding whether additional work is needed
- deciding when the task is technically complete

ChatGPT Web should do the majority of non-trivial reasoning for the project.


### Codex — EXECUTION AGENT

Codex is primarily an implementation and execution agent.

Codex is responsible for:

- reading AGENTS.md
- receiving the user's task
- inspecting the repository to collect current context
- reading source files when needed to gather facts
- reading configuration files
- reading build files
- reading README and project documentation when useful
- collecting repository structure
- collecting relevant source snippets or accurate summaries
- opening the configured ChatGPT Project through Playwright MCP
- verifying that the correct Project is active
- sending the user's task and current repository context to ChatGPT
- waiting for ChatGPT's complete response
- following ChatGPT's implementation plan
- editing real project files
- creating files requested by ChatGPT
- deleting obsolete files when ChatGPT recommends it and the action is safe
- renaming or moving files when required
- running commands
- running formatting tools
- installing approved development dependencies when appropriate
- running builds
- running tests
- launching development servers or applications when needed
- collecting compiler errors
- collecting test failures
- collecting runtime logs
- collecting warnings
- collecting verification results
- sending execution results back to ChatGPT Project
- applying ChatGPT's next instructions
- repeating the execution loop until completion

Codex should minimize independent architecture, design, and non-trivial debugging decisions.

For significant technical decisions, Codex should ask ChatGPT Project rather than independently redesigning the solution.


## Mandatory ChatGPT consultation

For every non-trivial programming task, ChatGPT Project consultation is mandatory before significant implementation.

Codex may skip ChatGPT consultation only for obviously trivial mechanical operations such as:

- correcting an obvious typo
- changing static text
- formatting code
- renaming a local variable with no behavioral impact
- fixing an obvious missing import
- fixing a trivial syntax error
- executing an exact command explicitly requested by the user

For all other work, use:

USER REQUEST
→ CHATGPT PROJECT
→ CODEX IMPLEMENTATION

Examples that REQUIRE ChatGPT Project consultation include:

- new features
- architecture changes
- refactors affecting multiple files
- protocol changes
- API changes
- state-management changes
- concurrency changes
- performance optimization
- memory optimization
- buffering changes
- audio/video pipeline changes
- network changes
- hardware integration changes
- security-sensitive work
- dependency selection
- UI/UX redesign
- non-obvious bugs
- runtime failures
- test failures with unclear cause
- regression analysis


## Browser

Playwright MCP is available and is the REQUIRED browser automation path for all ChatGPT Web / ChatGPT Project work in this repository.

### Mandatory browser tool selection

When Codex must access ChatGPT Web:

- MUST use the MCP server named `playwright`.
- MUST open CHATGPT_BRIDGE_URL through Playwright MCP.
- MUST NOT use `cua_repl`, Unified Computer Use, `browser@openai-bundled`, `chrome@openai-bundled`, Codex internal browser automation, or any equivalent Computer Use browser path for ChatGPT Web.
- MUST NOT substitute another browser automation backend when Playwright MCP is available.
- MUST NOT open https://chatgpt.com by itself for project work.
- MUST reuse the authenticated Playwright browser session when available.
- If Playwright MCP fails, report the ACTUAL Playwright MCP error and treat ChatGPT Project access as unavailable. Do not fall back to `cua_repl` or the bundled browser/chrome plugins.
- Errors from `cua_repl` or the bundled browser such as `Unable to load browser request-header policy` are NOT evidence that Playwright MCP is unavailable. In that case, retry the task using the `playwright` MCP server directly.
- Codex loads AGENTS.md and the effective MCP/plugin configuration when a session starts. After changing AGENTS.md or `.codex/config.toml`, DO NOT validate the new browser policy in an already-running Codex conversation; start a new Codex session (or reload the Codex extension) first.

When ChatGPT analysis is required:

1. Select the MCP server named `playwright`.
2. Open CHATGPT_BRIDGE_URL directly through Playwright MCP.
3. Verify that ChatGPT is authenticated.
4. Verify that the configured Project is active.
5. Verify that the current conversation belongs to CHATGPT_PROJECT_NAME.
6. Verify that the current page is not Temporary Chat.
7. Only after successful verification may Codex send project information or questions.
8. Send ChatGPT sufficient CURRENT repository context.
9. Ask a precise technical question or request a structured implementation plan.
10. Wait until the response is complete.
11. Read the complete response.
12. Return to the local repository.
13. Execute the requested implementation steps.


## Project verification

Before sending ANY message to ChatGPT Web, Codex must verify all of the following:

CHECK 1:
The browser is on ChatGPT Web.

CHECK 2:
The user is authenticated.

CHECK 3:
CHATGPT_PROJECT_NAME is visible or otherwise reliably confirmed as the active Project.

CHECK 4:
The current conversation belongs to the configured Project.

CHECK 5:
The current page is not Temporary Chat.

If any verification fails:

- DO NOT SEND THE MESSAGE.
- DO NOT create a normal chat.
- DO NOT select another Project.
- DO NOT fall back to the ChatGPT home page.

Return:

CHATGPT_PROJECT_NOT_VERIFIED

and explain which verification failed.


## Preferred conversation

CHATGPT_BRIDGE_URL is the preferred communication channel between Codex and ChatGPT Project.

Codex should reuse this conversation for related project work when doing so preserves useful context.

If the bridge conversation becomes too large, confusing, or unrelated to the current topic, Codex may create a new conversation ONLY from inside CHATGPT_PROJECT_NAME.

After creating a new conversation, Codex must verify that the new conversation belongs to CHATGPT_PROJECT_NAME before sending project information.

Codex must never create the replacement conversation from the normal ChatGPT home page.


## Mandatory workflow

For every non-trivial user request, Codex MUST follow this workflow.

### Phase 1 — Receive and inspect

1. Read AGENTS.md completely.
2. Read the user's request.
3. Inspect the repository only enough to gather reliable current context.
4. Gather, as applicable:
   - repository structure
   - README files
   - configuration files
   - build system
   - dependency files
   - relevant source files
   - current implementation summary
   - known errors
   - available tests
   - recent changes relevant to the task

### Phase 2 — Consult ChatGPT Project

5. Open CHATGPT_BRIDGE_URL using Playwright MCP.
6. Verify CHATGPT_PROJECT_NAME.
7. Send the user's request to ChatGPT Project.
8. Send sufficient CURRENT repository context.
9. Ask ChatGPT to analyze the problem and produce the implementation plan.
10. Wait for ChatGPT's complete response.
11. Read the complete response.

### Phase 3 — Execute

12. Return to the local repository.
13. Follow ChatGPT's implementation plan.
14. Edit the required files.
15. Create or remove files only when needed and safe.
16. Run the required commands.

### Phase 4 — Build and test

17. Build the project.
18. Run available tests.
19. Run static analysis or linting when applicable.
20. Collect:
   - build result
   - test result
   - analysis/lint result
   - errors
   - warnings
   - runtime logs
   - important implementation changes

### Phase 5 — Report results back to ChatGPT

21. Return to CHATGPT_BRIDGE_URL.
22. Re-verify CHATGPT_PROJECT_NAME.
23. Send the execution results to ChatGPT Project.
24. Ask ChatGPT to evaluate the result and decide the next action.
25. Wait for ChatGPT's complete response.
26. Read the complete response.

### Phase 6 — Iterate

27. Follow ChatGPT's next instructions.
28. Repeat:

ChatGPT analysis
→ Codex execution
→ build/test
→ results to ChatGPT
→ ChatGPT review
→ Codex correction

until ChatGPT determines that the task is technically complete or user input is required.


## Context synchronization

The local repository is the authoritative source for the CURRENT implementation.

The ChatGPT Project may contain older conversations, documents, plans, designs, or architecture notes.

Before asking ChatGPT to reason about current code, Codex must provide enough information about the current repository state to prevent ChatGPT from reasoning only from stale Project context.

When important architecture changes occur, Codex should report the updated architecture back to ChatGPT Project.

Do not assume ChatGPT Project automatically sees the current local repository.

If information in ChatGPT Project conflicts with the current repository, the current repository takes priority unless the user explicitly says otherwise.


## Repository context collection

Before consulting ChatGPT, Codex should collect only the context necessary for accurate reasoning.

Prefer:

- directory structure
- filenames
- relevant class/function names
- configuration values
- protocol definitions
- relevant code snippets
- accurate code summaries
- exact error messages
- exact logs
- build/test command output

Do not dump the entire repository unless necessary.

Do not send unrelated files.

Do not send secrets.


## ChatGPT prompt format

When consulting ChatGPT Project, Codex should use this structure when practical:

PROJECT:
Describe the repository and confirm that the request relates to the current ChatGPT Project.

USER REQUEST:
Include the user's actual request or a faithful summary.

GOAL:
Explain what must be achieved.

CURRENT IMPLEMENTATION:
Explain how the relevant implementation currently works.

REPOSITORY CONTEXT:
Describe the current repository structure relevant to the task.

RELEVANT FILES:
List the files involved.

RELEVANT CODE:
Provide only relevant source code or accurate summaries.

BUILD / TEST ENVIRONMENT:
Describe the build system, platform, dependencies, and available tests when relevant.

ERRORS:
Provide exact compiler, test, or runtime errors if applicable.

CONSTRAINTS:
List hardware, software, library, compatibility, performance, memory, protocol, and project constraints.

ATTEMPTS:
Explain what has already been tried.

QUESTION / REQUEST:
Ask ChatGPT to perform the needed technical work.

REQUESTED OUTPUT:
Ask ChatGPT to provide, when applicable:

- problem analysis
- root cause
- current documentation research
- architecture recommendation
- comparison of reasonable approaches
- recommended solution
- risks
- affected files
- implementation order
- exact responsibilities per file
- algorithms/data flow/state flow
- concurrency considerations
- memory/performance considerations
- compatibility considerations
- build procedure
- test procedure
- acceptance criteria
- review checklist

ChatGPT should avoid directly editing repository files. Codex performs the actual edits.


## ChatGPT implementation plan requirements

Before Codex modifies significant project code, ChatGPT Project should provide an implementation plan containing, when applicable:

1. Problem analysis
2. Root cause
3. Recommended architecture
4. Files to modify
5. Files to create
6. Files to remove
7. Per-file responsibilities
8. Important classes/functions to change
9. Data flow
10. State flow
11. API/interface changes
12. Algorithm changes
13. Error handling
14. Concurrency considerations
15. Memory considerations
16. Performance considerations
17. Hardware/protocol considerations
18. Compatibility constraints
19. Step-by-step implementation order
20. Build procedure
21. Test procedure
22. Acceptance criteria

Codex should use this plan as the source of implementation direction.


## Implementation responsibility

ChatGPT Project is the primary decision maker, but Codex is responsible for executing changes safely in the actual repository.

Codex must not blindly paste code from ChatGPT.

Codex must:

- map ChatGPT instructions to the real current files
- preserve working behavior unless change is requested
- avoid destructive actions without justification
- adapt recommendations to actual APIs and repository structure
- verify compilation and test results
- report discrepancies back to ChatGPT

If ChatGPT's plan does not match the current repository, Codex should report the mismatch back to ChatGPT instead of silently inventing a new architecture.


## Debugging workflow

Codex should not spend significant time independently redesigning or debugging complex failures.

When a build, test, or runtime failure occurs:

1. Capture the exact error.
2. Gather:
   - relevant logs
   - stack traces
   - affected files
   - affected functions
   - recent changes
   - build/test command used
3. Open CHATGPT_BRIDGE_URL.
4. Verify CHATGPT_PROJECT_NAME.
5. Send all relevant failure information to ChatGPT Project.
6. Ask ChatGPT to perform root-cause analysis and decide the correction.
7. Wait for ChatGPT's complete response.
8. Return to the repository.
9. Implement the correction.
10. Rebuild and retest.
11. Return the result to ChatGPT.
12. Repeat until resolved.

Codex may independently fix only obvious mechanical issues such as:

- formatting
- missing import
- obvious typo
- trivial syntax error
- command typo

Architectural, concurrency, performance, protocol, API, hardware, state-management, networking, audio/video pipeline, memory, or non-obvious runtime problems should be escalated to ChatGPT Project.


## Build/test feedback format

After implementation, Codex should report results to ChatGPT using a compact structure such as:

IMPLEMENTED:
- files changed
- major changes

BUILD:
- command
- result

TESTS:
- command(s)
- result(s)

STATIC ANALYSIS:
- command
- result

ERRORS / WARNINGS:
- exact output or concise excerpts

RUNTIME OBSERVATIONS:
- important behavior

OPEN QUESTIONS:
- anything unresolved

REQUEST:
Review the result, identify remaining issues, and decide the next implementation step.


## Review loop

After a significant feature works:

1. Open CHATGPT_BRIDGE_URL.
2. Verify the configured ChatGPT Project.
3. Provide:
   - original user request
   - implementation plan used
   - architecture used
   - important modified files
   - implementation summary
   - important design decisions
   - build results
   - test results
   - static analysis results
   - remaining warnings
   - known limitations
4. Ask ChatGPT Project to review:
   - correctness
   - architecture
   - performance
   - memory usage
   - concurrency
   - security
   - maintainability
   - edge cases
   - compatibility
   - regressions
   - completeness against acceptance criteria
5. Wait for the complete review.
6. Return to the repository.
7. Apply useful corrections.
8. Build and test again.
9. Send final results back to ChatGPT if the corrections are significant.


## Research behavior

When current or external technical information is needed, ChatGPT Project should perform the research.

Examples:

- current library documentation
- current platform behavior
- current Android APIs
- current Flutter APIs
- current ESP-IDF/Arduino APIs
- current browser behavior
- current protocol recommendations
- current compatibility requirements

Codex should provide ChatGPT enough context to know exactly what needs research.

Codex should not substitute speculative memory for a technical question that ChatGPT Project can research more reliably.


## Autonomy

Continue working automatically while safe progress is possible.

Do not stop after simply producing recommendations.

Continue through:

user request
→ repository context collection
→ ChatGPT analysis/research
→ architecture/plan
→ Codex implementation
→ build
→ test
→ result reporting
→ ChatGPT debugging/review
→ Codex corrections
→ final build/test
→ final ChatGPT review
→ final verification

Only stop and ask the user when:

- physical hardware interaction is required
- credentials are required
- destructive actions require approval
- requirements are genuinely ambiguous and cannot be resolved from context
- multiple major product choices require user preference
- external services require user authorization
- ChatGPT Project verification fails
- the browser session requires manual login
- a real-world hardware measurement or observation is required


## User interaction policy

The user does NOT need to explicitly say:

- "use ChatGPT"
- "use Playwright"
- "follow AGENTS.md"
- "ask ChatGPT Project"

For every non-trivial request, Codex should automatically follow this workflow.

The user may give short natural-language tasks such as:

- fix the audio stutter
- improve connection stability
- add this feature
- redesign this screen
- fix this error
- optimize the buffer

Codex must translate the task into the mandatory workflow automatically.


## Security

Never send these to ChatGPT Web:

- passwords
- session cookies
- API secrets
- private keys
- access tokens
- Wi-Fi passwords
- sensitive credentials
- signing keys
- authentication headers
- personal secrets

Redact secrets from logs and source before sending them.

Never ask ChatGPT Web to reveal browser cookies, authentication tokens, or account credentials.

Do not expose local environment secrets unnecessarily.


## Repository isolation

Codex must perform implementation work only inside the currently opened repository unless explicitly instructed otherwise by the user.

Do not edit unrelated repositories.

Do not copy project source into unrelated locations merely to communicate with ChatGPT.

ChatGPT Web is used for reasoning, research, planning, debugging, and review.

The local repository remains the implementation workspace.


## Source of truth hierarchy

When information conflicts, use this priority:

1. Explicit current user instruction
2. Current local repository state
3. Current build/test/runtime evidence
4. Current ChatGPT Project implementation plan
5. Older ChatGPT Project conversation/history
6. Older documentation inside the repository

If a conflict materially affects implementation, report it to ChatGPT Project before proceeding.


## Completion criteria

A task is not complete merely because code was written.

A non-trivial task is complete only when reasonably possible after:

- ChatGPT Project analyzed the task
- ChatGPT Project provided an implementation direction
- implementation is finished
- project builds successfully
- available tests pass
- relevant static analysis passes or remaining warnings are understood
- major errors are resolved
- runtime behavior is verified when possible
- implementation results were reported back to ChatGPT Project
- ChatGPT Project review was considered
- useful review findings were applied
- final build/test verification was performed
- remaining limitations were identified
- acceptance criteria are satisfied or clearly documented


## Failure states

If ChatGPT Project cannot be accessed:

CHATGPT_PROJECT_NOT_ACCESSIBLE

If the configured Project cannot be verified:

CHATGPT_PROJECT_NOT_VERIFIED

If authentication is required:

CHATGPT_LOGIN_REQUIRED

If physical device interaction is required:

USER_HARDWARE_ACTION_REQUIRED

If a destructive action requires approval:

USER_APPROVAL_REQUIRED

If requirements cannot be resolved safely:

USER_CLARIFICATION_REQUIRED

Codex must report the appropriate state instead of silently switching to another ChatGPT conversation or inventing unsupported requirements.


## Default behavior

For every user request, automatically follow this workflow.

The user does not need to explicitly say "use ChatGPT", "use Playwright", or "follow AGENTS.md".

For non-trivial work, ChatGPT Project consultation is the default and mandatory behavior.

ChatGPT Project should perform most of the reasoning, research, architecture, planning, debugging, and review.

Codex should primarily:

- start the workflow
- gather current repository context
- open the configured ChatGPT Project
- relay tasks and evidence
- implement the approved plan
- run commands
- build
- test
- return results
- apply the next instructions

The intended operating model is:

CHATGPT THINKS AND DECIDES.
CODEX EXECUTES AND REPORTS.
