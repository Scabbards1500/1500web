import { fromHtml } from "hast-util-from-html";
import { find, html } from "property-information";
import type { Nodes, Root } from "hast";

const VOID_ELEMENTS = new Set([
  "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr",
]);

function escapeHtml(value: string): string {
  return value.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

// Serialize the Markdown-generated nodes alongside raw HTML before parsing the
// whole fragment. Parsing individual raw nodes would lose paired inline spans
// whose contents include Markdown emphasis, links, or math.
function serialize(node: Nodes): string {
  switch (node.type) {
    case "root":
      return node.children.map(serialize).join("");
    case "text":
      return escapeHtml(node.value);
    case "raw":
      return node.value;
    case "comment":
      return `<!--${node.value}-->`;
    case "element": {
      const attributes = Object.entries(node.properties).flatMap(([key, value]) => {
        if (value === null || value === undefined || value === false) return [];
        const info = find(html, key);
        if (value === true && (info.boolean || info.overloadedBoolean)) return [info.attribute];
        const text = Array.isArray(value) ? value.join(info.commaSeparated ? "," : " ") : String(value);
        return [`${info.attribute}="${escapeHtml(text)}"`];
      });
      const opening = `<${node.tagName}${attributes.length ? ` ${attributes.join(" ")}` : ""}>`;
      return VOID_ELEMENTS.has(node.tagName)
        ? opening
        : `${opening}${node.children.map(serialize).join("")}</${node.tagName}>`;
    }
    default:
      return "";
  }
}

/** Parse note HTML with the installed HTML5 parser, preserving escaped code. */
export default function rehypeNoteHtml() {
  return (tree: Root): Root => fromHtml(serialize(tree), { fragment: true });
}
