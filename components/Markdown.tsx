import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import remarkBreaks from "remark-breaks";
import remarkCjkFriendly from "remark-cjk-friendly";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import rehypeSlug from "rehype-slug";
import { normalizeObsidian } from "@/lib/markdown";
import ArticleImage from "./ArticleImage";

type MarkdownProps = {
  content: string;
  /** 文章所在目录（相对 content/），用于解析相对路径的图片 */
  assetDir: string;
};

function resolveAsset(src: string, assetDir: string): string {
  if (/^(https?:)?\/\//.test(src) || src.startsWith("/") || src.startsWith("data:")) {
    return src;
  }
  const segments = [...assetDir.split("/"), ...decodeURI(src).split("/")];
  return `/api/content/${segments.map(encodeURIComponent).join("/")}`;
}

export default function Markdown({ content, assetDir }: MarkdownProps) {
  return (
    <div className="prose prose-slate max-w-none prose-headings:scroll-mt-24 prose-pre:bg-slate-900 prose-pre:text-slate-100 prose-code:before:content-none prose-code:after:content-none prose-img:my-0">
      <ReactMarkdown
        // remarkBreaks：Obsidian 默认把单个换行当作换行，这里保持一致
        // remarkCjkFriendly：让 `**粗体)**中文` 这类紧贴中文的强调也能生效
        remarkPlugins={[remarkGfm, remarkCjkFriendly, remarkMath, remarkBreaks]}
        // rehypeSlug 放在最前，保证标题 id 与 extractToc 生成的一致
        rehypePlugins={[rehypeSlug, rehypeKatex, [rehypeHighlight, { detect: false, ignoreMissing: true }]]}
        components={{
          img: ({ src, alt }) => (
            <ArticleImage src={typeof src === "string" ? resolveAsset(src, assetDir) : undefined} alt={alt} />
          ),
          a: ({ href, children }) => {
            const external = href && /^https?:\/\//.test(href);
            return (
              <a href={href} {...(external ? { target: "_blank", rel: "noopener noreferrer" } : {})}>
                {children}
              </a>
            );
          },
          table: ({ children }) => (
            <div className="overflow-x-auto">
              <table>{children}</table>
            </div>
          ),
        }}
      >
        {normalizeObsidian(content)}
      </ReactMarkdown>
    </div>
  );
}
