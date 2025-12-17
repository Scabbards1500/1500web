import Link from "next/link";
import SiteLayout from "@/components/SiteLayout";
import PostList from "@/components/PostList";
import { getPosts } from "@/lib/posts";

export default function Home() {
  const posts = getPosts();
  const featuredPosts = posts.slice(0, 2);

  return (
    <SiteLayout active="Homepage">
      <section className="mx-auto flex max-w-4xl flex-col gap-6 px-6 pt-12 md:px-12">
        <span className="text-xs font-semibold uppercase tracking-[0.35em] text-slate-400">
          Welcome
        </span>
        <h2 className="text-4xl font-semibold text-slate-900">Hello, I'm Hanling Wang</h2>
      </section>

      <section className="mx-auto flex max-w-4xl flex-col gap-6 px-6 pt-16 md:px-12">

        <PostList posts={featuredPosts} />
      </section>
    </SiteLayout>
  );
}
