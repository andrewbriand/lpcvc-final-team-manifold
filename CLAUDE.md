# LPCVC 2026 Track 1 (Team Manifold) — Claude Instructions

## Wiki vault

The persistent research wiki lives **outside** this codebase at `/Users/jrauvola/Desktop/wiki/` (private remote: `jrauvola/personal_wiki`). It hosts knowledge for multiple projects.

This codebase is bound to the wiki project slug **`lpcvc-2026-track1`** via the `.wiki-project` file at this directory's root.

## Wiki path resolution (override the plugin's defaults)

The `claude-obsidian` plugin's skills (`wiki`, `wiki-query`, `wiki-ingest`, `wiki-lint`, `save`, `autoresearch`, `canvas`) reference paths like `wiki/hot.md`. Use these instead:

**Vault root:** `/Users/jrauvola/Desktop/wiki/`

**Active project (resolved from this codebase):** `lpcvc-2026-track1`

**Workspace surfaces** — read these when the plugin's skill says to read `wiki/<surface>.md`:
- `wiki/hot.md` → `/Users/jrauvola/Desktop/wiki/projects/lpcvc-2026-track1/hot.md`
- `wiki/index.md` → `/Users/jrauvola/Desktop/wiki/projects/lpcvc-2026-track1/index.md`
- `wiki/log.md` → `/Users/jrauvola/Desktop/wiki/projects/lpcvc-2026-track1/log.md`
- `wiki/overview.md` → `/Users/jrauvola/Desktop/wiki/projects/lpcvc-2026-track1/overview.md`
- `wiki/experiments.md` → `/Users/jrauvola/Desktop/wiki/projects/lpcvc-2026-track1/experiments.md`

**Knowledge surfaces** (cross-project, unchanged) — read these at vault root:
- `wiki/concepts/`, `wiki/entities/`, `wiki/sources/`, `wiki/ideas/`, `wiki/questions/`, `wiki/comparisons/`, `wiki/canvas/`, `wiki/meta/`

**Always-loaded project index:** `/Users/jrauvola/Desktop/wiki/WORKSPACE.md`
**Vault-wide rules and schema:** `/Users/jrauvola/Desktop/wiki/CLAUDE.md`

## When writing back to the wiki

- New experiment notes → `/Users/jrauvola/Desktop/wiki/projects/lpcvc-2026-track1/experiments/<slug>.md`
- New concept/entity/source pages → vault root with `projects: [{slug: lpcvc-2026-track1, relevance: …}]` in frontmatter
- Hot cache updates → overwrite `/Users/jrauvola/Desktop/wiki/projects/lpcvc-2026-track1/hot.md` (keep under 500 words)
- Log entries → top of `/Users/jrauvola/Desktop/wiki/projects/lpcvc-2026-track1/log.md`

## Recommended reading order when querying the wiki for project context

1. `/Users/jrauvola/Desktop/wiki/projects/lpcvc-2026-track1/hot.md`
2. `/Users/jrauvola/Desktop/wiki/projects/lpcvc-2026-track1/index.md`
3. `/Users/jrauvola/Desktop/wiki/projects/lpcvc-2026-track1/overview.md`
4. `/Users/jrauvola/Desktop/wiki/meta/projects/REGISTRY.md`
5. Individual `concepts/`, `entities/`, `sources/` pages as needed
