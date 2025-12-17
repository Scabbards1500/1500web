import Link from "next/link";
import SiteLayout from "@/components/SiteLayout";
import { getInternships } from "@/lib/posts";
import PostList from "@/components/PostList";

export default function InternshipPage() {
  const internships = getInternships();

  return (
    <SiteLayout active="Internship">
      <section className="mx-auto flex max-w-4xl flex-col gap-4 px-6 pt-12 md:px-12">
        <span className="text-xs font-semibold uppercase tracking-[0.35em] text-slate-400">
          Internship
        </span>
        <h2 className="text-3xl font-semibold text-slate-900">合作与实践</h2>
        <p className="text-sm text-slate-600">
          将研究经验转化为工程价值，持续在真实环境中迭代验证。所有条目均由 `content/internship` 目录内的 Markdown 文件驱动。
        </p>
      </section>

      <section className="mx-auto max-w-4xl px-6 pt-12 md:px-12">
        {internships.length > 0 ? (
          <PostList posts={internships} headingLevel="h2" showSummary={false} contentType="internship" />
        ) : (
          <div className="rounded-3xl border border-dashed border-slate-300 bg-white/70 px-6 py-10 text-center text-sm text-slate-600">
            暂无实习经历，先在 `content/internship` 创建一个文件夹并放入 `index.md` 吧。
          </div>
        )}
      </section>

    </SiteLayout>
  );
}

