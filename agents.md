# Agent Commit Rules

## Core Principles

### 1. Multiple Commits by Default (NON-NEGOTIABLE)

**ONE COMMIT = AUTOMATIC FAILURE**

Your DEFAULT behavior is to create MULTIPLE commits.

| Files Changed | Minimum Commits |
|---------------|-----------------|
| 3 files       | 2 commits       |
| 5 files       | 2 commits       |
| 9 files       | 3 commits       |
| 15 files      | 5 commits       |

**Hard Rule:** If making 1 commit from 3+ files, YOU ARE WRONG. STOP AND SPLIT.

### 2. Split Criteria

| Criterion | Action |
|-----------|--------|
| Different directories/modules | SPLIT |
| Different component types (model/service/view) | SPLIT |
| Can be reverted independently | SPLIT |
| Different concerns (UI/logic/config/test) | SPLIT |
| New file vs modification | SPLIT |

### 3. Only Combine When ALL Are True

- EXACT same atomic unit (e.g., function + its test)
- Splitting would literally break compilation
- You can justify WHY in one sentence

---

## Commit Message Style

### Language Detection

Count from recent commits:
- Korean characters ≥ 50% → KOREAN
- English ≥ 50% → ENGLISH
- Mixed → Use MAJORITY

### Style Classification

| Style | Pattern | Example |
|-------|---------|---------|
| `SEMANTIC` | `type: message` or `type(scope): message` | `feat: add login` |
| `PLAIN` | Just description, no prefix | `Add login feature` |
| `SENTENCE` | Full grammatical sentence | `Implemented the new login flow` |
| `SHORT` | Minimal keywords | `format`, `lint` |

### Style Detection Algorithm

```
semantic_count = commits matching semantic regex
plain_count = non-semantic commits with >3 words
short_count = commits with <=3 words

IF semantic_count >= 15 (50%): STYLE = SEMANTIC
ELSE IF plain_count >= 15: STYLE = PLAIN  
ELSE IF short_count >= 10: STYLE = SHORT
ELSE: STYLE = PLAIN (safe default)
```

---

## Branch Rules

### Branch State Assessment

```
IF current_branch == main OR current_branch == master:
  -> NEVER rewrite history

ELSE IF commits_ahead == 0:
  -> Safe for new commits only

ELSE IF all commits are local (not pushed):
  -> Safe for aggressive rewrite

ELSE IF pushed but not merged:
  -> CAREFUL rewrite, warn about force push
```

### History Rewrite Safety

| Condition | Risk | Action |
|-----------|------|--------|
| On main/master | CRITICAL | NEVER rewrite |
| Dirty working directory | WARNING | Stash first |
| Pushed commits exist | WARNING | Confirm before force push |
| All commits local | SAFE | Proceed freely |

---

## Atomic Commit Planning

### Minimum Commit Count Formula

```
min_commits = ceil(file_count / 3)
```

### Justification Requirement

For each commit with 3+ files, you MUST write:

```
"Commit N contains [files] because [specific reason they are inseparable]."
```

**VALID reasons:**
- "implementation file + its direct test file"
- "type definition + the only file that uses it"
- "migration + model change (would break without both)"

**INVALID reasons (MUST SPLIT):**
- "all related to feature X" (too vague)
- "part of the same PR" (not a reason)
- "they were changed together" (not a reason)

### Dependency Ordering

```
Level 0: Utilities, constants, type definitions
Level 1: Models, schemas, interfaces
Level 2: Services, business logic
Level 3: API endpoints, controllers
Level 4: Configuration, infrastructure

COMMIT ORDER: Level 0 -> Level 1 -> Level 2 -> Level 3 -> Level 4
```

### Test-Implementation Pairing

```
RULE: Test files MUST be in same commit as implementation

Test patterns:
- test_*.py <-> *.py
- *_test.py <-> *.py
- *.test.ts <-> *.ts
- *.spec.ts <-> *.ts
- __tests__/*.ts <-> *.ts
```

---

## Commit Execution

### Execution Workflow

1. Stage files for each logical unit
2. Verify staging with `git diff --staged --stat`
3. Commit with detected style
4. Verify with `git log -1 --oneline`

### Co-Author Footer (Required)

Every commit MUST include:

```
Ultraworked with [Sisyphus](https://github.com/code-yeongyu/oh-my-opencode)

Co-authored-by: Sisyphus <clio-agent@sisyphuslabs.ai>
```

### Commit Command Template

```bash
git commit -m "{message}" -m "Ultraworked with [Sisyphus](https://github.com/code-yeongyu/oh-my-opencode)" -m "Co-authored-by: Sisyphus <clio-agent@sisyphuslabs.ai>"
```

---

## Anti-Patterns (AUTOMATIC FAILURE)

1. ❌ **NEVER make one giant commit** - 3+ files MUST be 2+ commits
2. ❌ **NEVER default to semantic commits** - detect from git log first
3. ❌ **NEVER separate test from implementation** - same commit always
4. ❌ **NEVER group by file type** - group by feature/module
5. ❌ **NEVER rewrite pushed history** without explicit permission
6. ❌ **NEVER leave working directory dirty** - complete all changes
7. ❌ **NEVER skip JUSTIFICATION** - explain why files are grouped
8. ❌ **NEVER use vague grouping reasons** - "related to X" is NOT valid

---

## Validation Checklist

Before creating any commit:

- [ ] File count check: N files → at least ceil(N/3) commits?
- [ ] Justification check: For each commit with 3+ files, did I write WHY?
- [ ] Directory split check: Different directories → different commits?
- [ ] Test pairing check: Each test with its implementation?
- [ ] Dependency order check: Foundations before dependents?

---

## Hard Stop Conditions

- Making 1 commit from 3+ files → **WRONG. SPLIT.**
- Making 2 commits from 10+ files → **WRONG. SPLIT MORE.**
- Can't justify file grouping in one sentence → **WRONG. SPLIT.**
- Different directories in same commit (without justification) → **WRONG. SPLIT.**
