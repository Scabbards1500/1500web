import Link from "next/link";
import type { Post } from "@/lib/posts";

function renderMarkdown(content: string) {
  const blocks = content
    .split(/\n{2,}/)
    .map((block) => block.trim())
    .filter(Boolean);

  return blocks.map((block, index) => {
    const lines = block.split("\n");
    const bulletList = lines.every((line) => /^[-*]\s+/.test(line));
    const orderedList = lines.every((line) => /^\d+\.\s+/.test(line));

    if (bulletList) {
      return (
        <ul key={index} className="list-disc pl-5 text-sm leading-6 text-slate-600">
          {lines.map((line, idx) => (
            <li key={idx}>{line.replace(/^[-*]\s+/, "")}</li>
          ))}
        </ul>
      );
    }

    if (orderedList) {
      return (
        <ol key={index} className="list-decimal pl-5 text-sm leading-6 text-slate-600">
          {lines.map((line, idx) => (
            <li key={idx}>{line.replace(/^\d+\.\s+/, "")}</li>
          ))}
        </ol>
      );
    }

    return (
      <p key={index} className="text-sm leading-7 text-slate-600">
        {block}
      </p>
    );
  });
}

type PostListProps = {
  posts: Post[];
  headingLevel?: "h2" | "h3";
  showSummary?: boolean;
};

export default function PostList({ posts, headingLevel = "h3", showSummary = true }: PostListProps) {
  return (
    <div className="flex flex-col gap-10">
      {posts.map((post) => (
        <article
          key={post.slug}
          className="flex flex-col gap-6 rounded-3xl border border-slate-200 bg-white p-6 shadow-sm transition-shadow hover:shadow-md md:flex-row"
        >
          <div className="flex flex-1 flex-col gap-4">
            <div className="space-y-2">
              {post.date && (
                <span className="text-xs font-medium uppercase tracking-widest text-slate-400">
                  {new Date(post.date).toLocaleDateString("zh-CN", {
                    year: "numeric",
                    month: "short",
                  })}
                </span>
              )}
              {headingLevel === "h2" ? (
                <h2 className="text-2xl font-semibold text-slate-900">{post.title}</h2>
              ) : (
                <h3 className="text-2xl font-semibold text-slate-900">{post.title}</h3>
              )}
              {post.subtitle && <p className="text-base text-slate-500">{post.subtitle}</p>}
            </div>

            {showSummary && post.summary && <p className="text-sm text-slate-500">{post.summary}</p>}

            <div className="space-y-4">{renderMarkdown(post.content)}</div>

            {post.links && (
              <div className="flex flex-wrap gap-3 pt-2 text-sm font-medium">
                {Object.entries(post.links).map(([key, url]) => (
                  <Link
                    key={key}
                    href={url}
                    className="rounded-full border border-slate-300 px-3 py-1 capitalize text-slate-600 transition hover:border-slate-900 hover:text-slate-900"
                  >
                    {key}
                  </Link>
                ))}
              </div>
            )}
          </div>

          {post.image && (
            <div className="md:w-64">
              <div className="h-48 overflow-hidden rounded-2xl border border-slate-200 bg-slate-100">
                <img
                  src={post.image}
                  alt={post.title}
                  className="h-full w-full object-cover"
                  loading="lazy"
                />
              </div>
            </div>
          )}
        </article>
      ))}
    </div>
  );
}

