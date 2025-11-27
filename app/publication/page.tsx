import SiteLayout from "@/components/SiteLayout";
import PostList from "@/components/PostList";
import { getPosts } from "@/lib/posts";

export default function PublicationPage() {
  const posts = getPosts();

  return (
    <SiteLayout active="Publication">
      <section className="mx-auto flex max-w-4xl flex-col gap-4 px-6 pt-12 md:px-12">
        <span className="text-xs font-semibold uppercase tracking-[0.35em] text-slate-400">
          Publication
        </span>
        <h2 className="text-3xl font-semibold text-slate-900">研究与项目成果</h2>
        <p className="text-sm text-slate-600">
          所有条目均由 `content/posts` 目录内的 Markdown 文件驱动，支持添加封面、摘要、链接等前置信息。
        </p>
      </section>

      <section className="mx-auto max-w-4xl px-6 pt-12 md:px-12">
        <PostList posts={posts} />
      </section>
    </SiteLayout>
  );
}

