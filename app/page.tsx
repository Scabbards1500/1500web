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
        <h2 className="text-4xl font-semibold text-slate-900">你好，我是 Hanling Wang</h2>
        <p className="text-base leading-7 text-slate-600">
          专注于传感网络、机器学习与城市韧性研究，热衷将前沿算法落地到真实场景中。
          这里记录我的研究动态、实习经历与项目背后的故事。
        </p>
        <div className="flex flex-wrap gap-3 text-sm">
          <Link
            href="/publication"
            className="rounded-full bg-slate-900 px-5 py-2 text-white transition hover:bg-slate-700"
          >
            查看全部成果
          </Link>
          <Link
            href="/gallery"
            className="rounded-full border border-slate-300 px-5 py-2 text-slate-600 transition hover:border-slate-900 hover:text-slate-900"
          >
            浏览影像记忆
          </Link>
        </div>
      </section>

      <section className="mx-auto flex max-w-4xl flex-col gap-6 px-6 pt-16 md:px-12">
        <header className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <h3 className="text-2xl font-semibold text-slate-900">最新项目摘录</h3>
            <Link href="/publication" className="text-sm font-semibold text-slate-500 hover:text-slate-900">
              查看更多
            </Link>
          </div>
          <p className="text-sm text-slate-500">
            以下内容来自 `content/posts` 目录下的 Markdown 文件，可按文件夹扩展。
          </p>
        </header>

        <PostList posts={featuredPosts} />
      </section>

      <section className="mx-auto grid max-w-4xl gap-6 px-6 py-16 md:grid-cols-3 md:px-12">
        <div className="flex flex-col gap-3 rounded-3xl border border-slate-200 bg-white p-6">
          <h4 className="text-lg font-semibold text-slate-900">Publication</h4>
          <p className="text-sm text-slate-600">
            系统整理论文、项目报告与典型案例分析，按年份归档检索。
          </p>
          <Link href="/publication" className="text-xs font-semibold uppercase tracking-widest text-slate-500 hover:text-slate-900">
            进入页面 →
          </Link>
        </div>

        <div className="flex flex-col gap-3 rounded-3xl border border-slate-200 bg-white p-6">
          <h4 className="text-lg font-semibold text-slate-900">Internship</h4>
          <p className="text-sm text-slate-600">
            汇总科研与企业合作经历，梳理关键成果与经验收获。
          </p>
          <Link href="/internship" className="text-xs font-semibold uppercase tracking-widest text-slate-500 hover:text-slate-900">
            进入页面 →
          </Link>
        </div>

        <div className="flex flex-col gap-3 rounded-3xl border border-slate-200 bg-white p-6">
          <h4 className="text-lg font-semibold text-slate-900">Gallery</h4>
          <p className="text-sm text-slate-600">
            记录实验现场、展览与旅途瞬间，展示不同维度的故事。
          </p>
          <Link href="/gallery" className="text-xs font-semibold uppercase tracking-widest text-slate-500 hover:text-slate-900">
            进入页面 →
          </Link>
        </div>
      </section>
    </SiteLayout>
  );
}
