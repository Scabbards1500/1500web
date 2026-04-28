import type { Post } from "@/lib/posts";

type HomeContentProps = {
  posts: Post[];
};

export default function HomeContent({ posts }: HomeContentProps) {
  const sectionOrder: Array<NonNullable<Post["contentType"]>> = ["research", "internship", "awards"];

  const sectionConfig: Record<NonNullable<Post["contentType"]>, { title: string; emptyText: string }> = {
    research: { title: "Research", emptyText: "No research projects yet." },
    internship: { title: "Internship", emptyText: "No internship experience yet." },
    awards: { title: "Awards", emptyText: "No awards listed yet." },
    notes: { title: "Notes", emptyText: "No notes yet." },
  };

  const groupedPosts = posts.reduce<Record<string, Post[]>>((acc, post) => {
    const key = post.contentType ?? "others";
    if (!acc[key]) {
      acc[key] = [];
    }
    acc[key].push(post);
    return acc;
  }, {});

  const formatDate = (date?: string) => {
    if (!date) {
      return null;
    }
    return new Date(date).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
    });
  };

  return (
    <section className="mx-auto flex max-w-5xl flex-col gap-12 px-6 pb-16 pt-16 md:px-12">
      {sectionOrder.map((sectionKey) => {
        const sectionPosts = groupedPosts[sectionKey] ?? [];
        const config = sectionConfig[sectionKey];

        return (
          <section key={sectionKey} className="space-y-5">
            <h3 className="border-b border-slate-200 pb-2 text-3xl font-bold text-slate-900 md:text-4xl">
              {config.title}
            </h3>

            {sectionPosts.length === 0 ? (
              <p className="text-sm text-slate-500">{config.emptyText}</p>
            ) : (
              <div className="space-y-6">
                {sectionPosts.map((post) => (
                  <article key={post.slug} className="space-y-2 border-b border-slate-200 pb-5 last:border-none">
                    <div className="relative pr-28">
                      <h4 className="text-base font-semibold text-slate-900">{post.title}</h4>
                      {formatDate(post.date) && (
                        <span className="absolute right-0 top-0 rounded-md bg-sky-50 px-2 py-1 text-xs font-semibold uppercase tracking-wider text-sky-700">
                          {formatDate(post.date)}
                        </span>
                      )}
                    </div>

                    {post.author && <p className="mt-1 text-sm text-slate-600">{post.author}</p>}
                    {sectionKey === "internship" && post.subtitle && (
                      <p className="text-sm text-sky-700">{post.subtitle}</p>
                    )}
                    {post.status && (
                      <p className="text-sm text-slate-700">
                        <span className="font-medium text-emerald-700">Status:</span> {post.status}
                      </p>
                    )}
                    {post.links && Object.keys(post.links).length > 0 && (
                      <div className="flex flex-wrap gap-3 pt-1 text-sm">
                        {Object.entries(post.links).map(([key, url]) => (
                          <a
                            key={key}
                            href={url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-sky-700 underline underline-offset-4 transition hover:text-sky-900"
                          >
                            {key}
                          </a>
                        ))}
                      </div>
                    )}
                  </article>
                ))}
              </div>
            )}
          </section>
        );
      })}
    </section>
  );
}

