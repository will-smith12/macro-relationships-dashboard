# Macro-Financial Relationships — MCP server for Claude Desktop

This lets you **chat with Claude Desktop about your analysis** and have it pull
**exact numbers on demand** from the exported results — the master summary, any
single relationship's full dossier, the cross-correlation function, the rolling
correlation, structural breaks, the data catalogue, or any individual metric.

## How it fits together

```
macro_relationships_master.ipynb   ──exports──►   *.csv   ──read by──►   macro_mcp_server.py
        (compute layer)                          (data)                   (MCP query layer)
                                                                                │ stdio
                                                                          Claude Desktop
```

The notebook stays the single compute layer. The MCP server is a thin, read-only
query layer over the CSVs it exports. Claude Desktop launches the server as a
local subprocess (stdio) and calls its tools while you chat — no data leaves your
machine beyond your normal Claude conversation.

## One-command setup

```bash
./setup_mcp.sh
```

That creates an isolated `.venv-mcp`, installs the dependencies (`mcp`, `pandas`,
`tabulate`), and **safely merges** a `macro-relationships` entry into your existing
Claude Desktop config — preserving all your current settings and writing a
timestamped backup first.

Then **fully quit and reopen Claude Desktop** (⌘Q, relaunch) so it loads the
server. You should see `macro-relationships` appear in Claude's tools (the 🔌 /
tools menu).

### Manual install (alternative)

If you'd rather edit the config yourself, copy the block from
`claude_desktop_config.snippet.json` into:

```
~/Library/Application Support/Claude/claude_desktop_config.json
```

merging it under a top-level `"mcpServers"` key, then restart Claude Desktop.

## Available tools

| Tool | What Claude can ask for |
|------|--------------------------|
| `list_relationships` | The seven relationships (number + name). |
| `get_methodology` | The four supervising principles + ADF routing. |
| `get_master_summary` | The full one-row-per-relationship summary table. |
| `get_relationship` | A single relationship's complete dossier. |
| `get_cross_correlation` | r at every lag ±12Q, ±95% band, significant lags. |
| `get_rolling_correlation` | The 12-quarter rolling correlation over time. |
| `get_structural_breaks` | Detected break quarters for a relationship. |
| `get_data_catalogue` | Every series: source, range, frequency, n, I(d). |
| `search_metrics` | Free-text search across every computed metric. |

`get_relationship` (and the other per-relationship tools) accept a number (`"3"`),
a keyword (`"Okun"`, `"output gap"`, `"yield slope"`, `"FX"`), or the full label.

The saved figures are also exposed as `figure://<name>` resources (e.g.
`figure://fig_2_unemployment_vs_cpi`).

## Example prompts

- "List the macro relationships you can query."
- "Explain the Phillips curve result and why it's regime-dependent."
- "Which relationships flip sign across regimes?"
- "Show me the lead/lag structure for the yield-curve slope vs GDP — does the
  slope actually lead growth?"
- "For inflation vs the policy rate, does the link survive controlling for energy?"
- "Walk me through the methodology behind the headline correlations."

## Keeping it current

If you re-run the notebook, the CSVs update and the server serves the new numbers
the next time Claude starts it (it reads them at launch). To pick up fresh data
mid-session, just restart Claude Desktop.

## Troubleshooting

- **Server doesn't appear:** make sure you *fully quit* Claude Desktop (⌘Q) and
  reopened it; a window close is not enough.
- **"No analysis artefacts found":** run the notebook so the CSVs exist next to
  `macro_mcp_server.py`.
- **Reset the config:** restore the most recent `claude_desktop_config.json.bak-*`
  backup that `setup_mcp.sh` created.
