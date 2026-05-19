"""Utilities for part 02 / 01: tiny graph from the agreement and annex."""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape as html_escape
import json
from pathlib import Path
import re
import webbrowser

from dateutil import parser as date_parser
from pdfminer.high_level import extract_text
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelSettings
from typing import Optional

from graph_rag_workshop.utils.pydantic_utils import get_ollama_model
from graph_rag_workshop.utils.part_02_rag_utils import retrieve_context


SMALL_CONTRACT_SCHEMA = {
    "party": "A person or company with a role in the agreement.",
    "obligation": "Something a party must do.",
    "date": "A due date, event date, or notice date.",
}


class Party(BaseModel):
    role: str = Field(description="Contract role, for example Provider or Organizer.")
    name: str = Field(description="Company name.")


class Obligation(BaseModel):
    party: str = Field(description="The role that must do the action.")
    action: str = Field(description="What the party must do.")
    date: Optional[str] = Field(
        default=None, description="Due date/timing. Use null if not explicitly stated."
    )
    source_quote: str = Field(
        description="Very short quote supporting this obligation."
    )


class ContractGraphExtraction(BaseModel):
    parties: list[Party] = Field(default_factory=list)
    obligations: list[Obligation] = Field(default_factory=list)


@dataclass
class GraphNode:
    id: str
    kind: str
    label: str
    properties: dict[str, str] = field(default_factory=dict)


@dataclass
class GraphEdge:
    source: str
    relation: str
    target: str
    source_doc: str
    evidence: str


@dataclass
class KnowledgeGraph:
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: list[GraphEdge] = field(default_factory=list)
    edge_keys: set[tuple[str, str, str, str]] = field(default_factory=set)

    def add_node(
        self,
        node_id: str,
        kind: str,
        label: str,
        properties: dict[str, str] | None = None,
    ) -> None:
        self.nodes.setdefault(
            node_id, GraphNode(node_id, kind, label, properties or {})
        )

    def add_edge(
        self,
        source: str,
        relation: str,
        target: str,
        source_doc: str,
        evidence: str,
    ) -> None:
        key = (source, relation, target, source_doc)
        if key not in self.edge_keys:
            self.edge_keys.add(key)
            self.edges.append(GraphEdge(source, relation, target, source_doc, evidence))

    def merge(self, other: "KnowledgeGraph") -> None:
        for node in other.nodes.values():
            self.add_node(node.id, node.kind, node.label, node.properties)
        for edge in other.edges:
            self.add_edge(
                edge.source,
                edge.relation,
                edge.target,
                edge.source_doc,
                edge.evidence,
            )


def extract_facts(text: str) -> ContractGraphExtraction:
    schema_text = "\n".join(
        f"- {key}: {value}" for key, value in SMALL_CONTRACT_SCHEMA.items()
    )

    GRAPH_CONSTRUCTION_MODEL_SETTINGS = ModelSettings(
        thinking="minimal",
        temperature=0,
        max_tokens=700,
    )

    INSTRUCTIONS = (
        "Extract only the core contract facts. Return valid JSON matching the "
        "requested output type. Do not explain your answer.\n"
        "Use this graph schema:\n"
        f"{schema_text}\n\n"
        "Work in three passes:\n"
        "1. Parties: there are only two parties. Use the roles Organizer and "
        "Provider. Extract their company names from the agreement.\n"
        "2. Obligations: extract only the most important obligations for those "
        "two roles. Keep at most 8 obligations total.\n\n"
        "3. For each obligation, if present, include the related date. Confirm that the date is accurate.\n"
        "Return exactly this JSON shape:\n"
        "{\n"
        '  "parties": [\n'
        '    {"role": "Organizer", "name": "..."},\n'
        '    {"role": "Provider", "name": "..."}\n'
        "  ],\n"
        '  "obligations": [\n'
        '    {"party": "Organizer or Provider", "action": "...", "date": "...", "source_quote": "..."}\n'
        "  ]\n"
        "}\n\n"
        "Only use facts explicitly stated in the text. "
        "Every obligation must include party, action, date, and source_quote. "
        "Use null for date when no date is stated.\n"
        "Use a short exact quote for source_quote. Do not invent facts."
    )

    agent = Agent(
        model=get_ollama_model(),
        output_type=ContractGraphExtraction,
        instructions=INSTRUCTIONS,
        model_settings=GRAPH_CONSTRUCTION_MODEL_SETTINGS,
    )
    return agent.run_sync(text).output


def facts_to_graph(facts: ContractGraphExtraction, source_doc: str) -> KnowledgeGraph:
    graph = KnowledgeGraph()

    for party in facts.parties:
        role = party.role
        graph.add_node(f"party:{role}", "party", role, {"name": party.name})

    for obligation in facts.obligations:
        party_id = f"party:{obligation.party}"
        obligation_id = f"obligation:{obligation.action}"

        graph.add_node(party_id, "party", obligation.party)
        graph.add_node(obligation_id, "obligation", obligation.action)
        graph.add_edge(
            party_id, "must_do", obligation_id, source_doc, obligation.source_quote
        )

        if obligation.date:
            date_id = f"date:{obligation.date}"
            graph.add_node(date_id, "date", obligation.date)
            graph.add_edge(
                obligation_id, "due_by", date_id, source_doc, obligation.source_quote
            )

    return graph


def graph_edges_to_text(graph: KnowledgeGraph, edges: list[GraphEdge]) -> str:
    lines = []
    for edge in edges:
        source = graph.nodes[edge.source]
        target = graph.nodes[edge.target]
        lines.append(
            f"{source.label} --{edge.relation}--> {target.label}\n"
            f"Source: {edge.source_doc}\n"
            f"Evidence: {edge.evidence}"
        )
    return "\n\n".join(lines)


def graph_to_text(graph: KnowledgeGraph) -> str:
    lines = ["NODES"]
    for node in graph.nodes.values():
        props = ", ".join(
            f"{key}={value}" for key, value in node.properties.items() if value
        )
        suffix = f" ({props})" if props else ""
        lines.append(f"- {node.id} [{node.kind}] {node.label}{suffix}")
    lines.append("\nEDGES")
    lines.append(graph_edges_to_text(graph, graph.edges))
    return "\n".join(lines)


def extend_graph_from_retrieved_chunks(
    graph: KnowledgeGraph,
    chunks: str | list[str],
) -> KnowledgeGraph:
    """Merge facts extracted from RAG chunks into the existing graph."""
    chunks_text = (
        "\n\n------------\n\n".join(chunks) if isinstance(chunks, list) else chunks
    )
    graph.merge(facts_to_graph(extract_facts(chunks_text), "retrieved_chunks"))
    return graph


def retrieve_graph_context(
    graph: KnowledgeGraph,
    question: str,
    max_results: int = 12,
) -> str:
    """Return graph edges that overlap with the user question."""
    query_terms = set(re.findall(r"[a-z0-9]+", question.lower()))
    scored = []
    for edge in graph.edges:
        source = graph.nodes[edge.source]
        target = graph.nodes[edge.target]
        text = f"{source.label} {edge.relation} {target.label} {edge.evidence}".lower()
        score = len(query_terms & set(re.findall(r"[a-z0-9]+", text)))
        scored.append((score, edge))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [edge for score, edge in scored[:max_results] if score > 0]
    selected = selected or graph.edges[:max_results]
    return graph_edges_to_text(graph, selected)


def search_graph_rag_context(
    vector_database,
    graph_json_path: Path,
    question: str,
    max_results: int = 5,
) -> str:
    """Search vector chunks, expand the graph, and return both contexts."""
    vector_context = retrieve_context(vector_database, question)
    expanded_graph = extend_graph_from_retrieved_chunks(
        load_graph(graph_json_path),
        vector_context,
    )
    graph_context = retrieve_graph_context(
        expanded_graph,
        question,
        max_results=max_results,
    )
    return (
        "Schema graph facts:\n"
        f"{graph_context}\n\n"
        "Retrieved document chunks:\n"
        f"{vector_context}"
    )


def _summarize_text(text: str, max_words: int = 1) -> str:
    words = " ".join(text.split()).split(" ")
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + "..."


def _node_summary_label(node: GraphNode) -> str:
    base_text = node.label or node.id
    return _summarize_text(base_text)


def _format_date_label(text: str) -> str:
    try:
        parsed_date = date_parser.parse(text, fuzzy=True, dayfirst=True)
    except (ValueError, TypeError):
        return text.strip()
    if parsed_date.hour == 0 and parsed_date.minute == 0 and parsed_date.second == 0:
        return parsed_date.strftime("%d/%m/%y")
    else:
        return parsed_date.strftime("%H:%M")

def _node_expanded_label(node: GraphNode) -> str:
    lines = [node.label or node.id]
    lines.extend(
        f"{key}: {value}" for key, value in node.properties.items() if value
    )
    return "\n".join(lines)


def _node_title(node: GraphNode) -> str:
    parts = [f"<b>{html_escape(node.kind)}</b>", html_escape(node.label or node.id)]
    parts.extend(
        f"{html_escape(key)}: {html_escape(value)}"
        for key, value in node.properties.items()
        if value
    )
    return "<br>".join(parts)


def _inject_double_click_expander(output_path: Path) -> None:
    html = output_path.read_text(encoding="utf-8")
    if "function toggleNodeExpansion" in html:
        return

    script = """
    <script type="text/javascript">
    (function(){
        function toggleNodeExpansion(params) {
            if (!params.nodes || params.nodes.length !== 1) {
                return;
            }
            var nodeId = params.nodes[0];
            var node = nodes.get(nodeId);
            if (!node) {
                return;
            }

            var isExpanded = node.expanded === true;
            nodes.update({
                id: nodeId,
                label: isExpanded ? node.summaryLabel : node.expandedLabel,
                expanded: !isExpanded
            });
        }

        network.on("doubleClick", function (params) {
            toggleNodeExpansion(params);
        });
    })();
    </script>
    """

    html = html.replace("</body>", f"{script}\n    </body>")
    output_path.write_text(html, encoding="utf-8")


def visualize_graph(
    graph: KnowledgeGraph,
    output_path: Path,
    open_browser: bool = False,
) -> Path:
    from pyvis.network import Network

    network = Network(
        height="760px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#111827",
        directed=True,
        notebook=False,
        cdn_resources="remote",
    )
    network.force_atlas_2based(gravity=-50, central_gravity=0.005, spring_length=800)

    styles = {
        "party": {"color": "#90d1f2", "shape": "circle"},
        "obligation": {"color": "#f7b479", "shape": "circle"},
        "date": {"color": "#87e8ab", "shape": "circle"},
    }

    for node in graph.nodes.values():
        style = styles[node.kind]
        if getattr(node, "kind", None) == "date":
            summary_label = _format_date_label(node.label)
        else:
            summary_label = _node_summary_label(node)
        expanded_label = node.label or node.id
        network.add_node(
            node.id,
            label=summary_label,
            title=_node_title(node),
            color=style["color"],
            shape=style["shape"],
            summaryLabel=summary_label,
            expandedLabel=expanded_label,
            expanded=False,
        )

    for edge in graph.edges:
        network.add_edge(
            edge.source,
            edge.target,
            label=edge.relation,
            title=f"{edge.source_doc}<br>{edge.evidence}",
            arrows="to",
        )

    network.set_options(
        """
        {
            "nodes": {
                                "font": {"size": 24, "face": "Inter, Arial", "color": "white", "bold": {"color": "white"}},
                "borderWidth": 2,
                                "scaling": {"label": true, "min": 20, "max": 40}
            },
          "edges": {
            "font": {"size": 22, "align": "middle"},
            "color": {"color": "#9ca3af", "highlight": "#111827"},
                        "smooth": false
          },
          "physics": {
            "stabilization": {"iterations": 500},
            "barnesHut": {"damping": 0.8, "springLength": 300},
            "maxVelocity": 30
          },
          "interaction": {"hover": true, "navigationButtons": true, "keyboard": true, "dragNodes": true}
        }
        """
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    network.write_html(str(output_path), notebook=False, open_browser=False)
    _inject_double_click_expander(output_path)
    if open_browser:
        webbrowser.open(output_path.resolve().as_uri())
    return output_path


def save_graph(graph: KnowledgeGraph, save_path: Path) -> Path:
    """Save the graph built in part 02 / 01 for reuse in part 02 / 02."""
    data = {
        "nodes": [
            {
                "id": node.id,
                "kind": node.kind,
                "label": node.label,
                "properties": node.properties,
            }
            for node in graph.nodes.values()
        ],
        "edges": [
            {
                "source": edge.source,
                "relation": edge.relation,
                "target": edge.target,
                "source_doc": edge.source_doc,
                "evidence": edge.evidence,
            }
            for edge in graph.edges
        ],
    }
    save_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return save_path


def load_graph(save_path: Path) -> KnowledgeGraph:
    """Load the graph saved in save_path."""
    if not save_path.exists():
        raise FileNotFoundError(f"Saved graph not found at {save_path}. ")

    data = json.loads(save_path.read_text(encoding="utf-8"))
    graph = KnowledgeGraph()
    for node in data.get("nodes", []):
        graph.add_node(
            node["id"],
            node["kind"],
            node["label"],
            node.get("properties") or {},
        )
    for edge in data.get("edges", []):
        graph.add_edge(
            edge["source"],
            edge["relation"],
            edge["target"],
            edge["source_doc"],
            edge["evidence"],
        )
    return graph
