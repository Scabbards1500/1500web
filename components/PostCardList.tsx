import { CATEGORIES, postHref, type PostMeta } from "@/lib/posts";

type PostCardListProps = {
  posts: PostMeta[];
  showCategory?: boolean;
};

function formatDate(date?: string) {
  if (!date) return null;
  return date.replace(/-/g, ".");
}

export default function PostCardList({ posts, showCategory = false }: PostCardListProps) {
  return (
    <div className="flex flex-col gap-4">
      {posts.map((post, index) => (
        <a
          key={postHref(post)}
          href={postHref(post)}
          // 文章在新标签页打开，列表页保持不动
          target="_blank"
          rel="noopener noreferrer"
          className="group block rounded-2xl border border-slate-200 bg-white p-5 shadow-sm transition-all hover:-translate-y-0.5 hover:border-slate-300 hover:shadow-md opacity-0 animate-fade-in-up"
          style={{ animationDelay: `${index * 40}ms`, animationFillMode: "forwards" }}
        >
          <div className="flex items-start justify-between gap-4">
            <h3 className="text-lg font-semibold text-slate-900 group-hover:text-sky-700">{post.title}</h3>
            {post.date && (
              <span className="shrink-0 rounded-md bg-sky-50 px-2 py-1 text-xs font-semibold tracking-wider text-sky-700">
                {formatDate(post.date)}
              </span>
            )}
          </div>
          {post.abstract && <p className="mt-2 line-clamp-2 text-sm leading-6 text-slate-600">{post.abstract}</p>}
          {(showCategory || post.tags?.length) && (
            <div className="mt-3 flex flex-wrap gap-2 text-xs font-medium">
              {showCategory && (
                <span className="rounded-full bg-slate-900 px-2.5 py-0.5 text-white">
                  {CATEGORIES.find((c) => c.slug === post.category)?.label}
                  {post.series && ` · ${post.series}`}
                </span>
              )}
              {post.tags?.map((tag) => (
                <span key={tag} className="rounded-full bg-slate-100 px-2.5 py-0.5 text-slate-600">
                  {tag}
                </span>
              ))}
            </div>
          )}
        </a>
      ))}
    </div>
  );
}
