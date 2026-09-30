import Link from "next/link";
import SiteLayout from "@/components/SiteLayout";
import PostCardList from "@/components/PostCardList";
import { CATEGORIES, getAllPosts, toMeta } from "@/lib/posts";

const RECENT_COUNT = 8;

export default function Home() {
  const posts = getAllPosts().map(toMeta);
  const countByCategory = posts.reduce<Record<string, number>>((acc, post) => {
    acc[post.category] = (acc[post.category] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <SiteLayout active="Homepage">
      <section className="mx-auto flex max-w-5xl flex-col gap-6 px-6 pt-12 md:px-12">
        <span className="text-xs font-semibold uppercase tracking-[0.35em] text-slate-400">
          Welcome
        </span>
        <h2 className="text-4xl font-semibold text-slate-900">Hello, I'm Hanling Wang</h2>
        <p className="text-base text-slate-600">
          I am a researcher interested in building reliable and interpretable AI systems, with a focus on large language models. My research experience spans medical image segmentation, multimodal learning, and LLM reasoning. Moving forward, I aim to explore how structured representations and  reasoning can enhance the reliability of intelligent systems  :-)
        </p>
      </section>

      <section className="mx-auto grid max-w-5xl grid-cols-2 gap-4 px-6 pt-10 md:grid-cols-4 md:px-12">
        {CATEGORIES.map((category) => (
          <Link
            key={category.slug}
            href={`/blog/${category.slug}`}
            className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm transition hover:border-slate-900"
          >
            <p className="text-sm font-semibold text-slate-900">{category.label}</p>
            <p className="mt-1 text-2xl font-bold text-sky-700">{countByCategory[category.slug] ?? 0}</p>
            <p className="mt-1 text-xs text-slate-500">{category.description}</p>
          </Link>
        ))}
      </section>

      <section className="mx-auto max-w-5xl space-y-5 px-6 pb-16 pt-12 md:px-12">
        <h3 className="border-b border-slate-200 pb-2 text-3xl font-bold text-slate-900 md:text-4xl">Recent Posts</h3>
        {posts.length === 0 ? (
          <p className="text-sm text-slate-500">No posts yet.</p>
        ) : (
          <PostCardList posts={posts.slice(0, RECENT_COUNT)} showCategory />
        )}
      </section>
    </SiteLayout>
  );
}
