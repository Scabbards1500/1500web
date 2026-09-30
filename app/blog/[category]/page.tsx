import { notFound } from "next/navigation";
import SiteLayout from "@/components/SiteLayout";
import PostCardList from "@/components/PostCardList";
import { CATEGORIES, getCategory, getPostsByCategory, toMeta, type PostMeta } from "@/lib/posts";

export function generateStaticParams() {
  return CATEGORIES.map((c) => ({ category: c.slug }));
}

export async function generateMetadata({ params }: { params: Promise<{ category: string }> }) {
  const { category } = await params;
  return { title: `${getCategory(category)?.label ?? "Blog"} | Hanling Wang` };
}

export default async function CategoryPage({ params }: { params: Promise<{ category: string }> }) {
  const { category: slug } = await params;
  const category = getCategory(slug);
  if (!category) notFound();

  const posts = getPostsByCategory(slug).map(toMeta);

  // 按子文件夹分组，根目录下的文章放在最前
  const groups = posts.reduce<Map<string, PostMeta[]>>((acc, post) => {
    const list = acc.get(post.series) ?? [];
    list.push(post);
    acc.set(post.series, list);
    return acc;
  }, new Map());
  const orderedGroups = [...groups.entries()].sort(([a], [b]) => (a === "" ? -1 : b === "" ? 1 : a.localeCompare(b)));

  return (
    <SiteLayout active={category.label}>
      <section className="mx-auto flex max-w-5xl flex-col gap-4 px-6 pt-12 md:px-12">
        <span className="text-xs font-semibold uppercase tracking-[0.35em] text-slate-400">Blog</span>
        <h2 className="text-3xl font-semibold text-slate-900">{category.label}</h2>
        <p className="text-sm text-slate-600">
          {category.description} · {posts.length} 篇
        </p>
      </section>

      <section className="mx-auto flex max-w-5xl flex-col gap-10 px-6 pt-10 md:px-12 min-h-[calc(100vh-300px)]">
        {posts.length === 0 ? (
          <div className="rounded-3xl border border-dashed border-slate-300 bg-white/70 px-6 py-10 text-center text-sm text-slate-600">
            Under Construction :-)
          </div>
        ) : (
          orderedGroups.map(([series, list]) => (
            <div key={series || "_root"} className="space-y-4">
              {series && (
                <h3 className="border-b border-slate-200 pb-2 text-xl font-bold text-slate-900">{series}</h3>
              )}
              <PostCardList posts={list} />
            </div>
          ))
        )}
      </section>
    </SiteLayout>
  );
}
