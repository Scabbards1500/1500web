import Link from "next/link";
import SiteLayout from "@/components/SiteLayout";
import { getAwards } from "@/lib/posts";
import PostList from "@/components/PostList";

export default function AwardsPage() {
  const awards = getAwards();

  return (
    <SiteLayout active="Awards">
      <section className="mx-auto flex max-w-5xl flex-col gap-4 px-6 pt-12 md:px-12">
        <span className="text-xs font-semibold uppercase tracking-[0.35em] text-slate-400">
          Awards
        </span>
        <h2 className="text-3xl font-semibold text-slate-900">Awards</h2>
      </section>

      <section className="mx-auto max-w-5xl px-6 pt-12 md:px-12">
        {awards.length > 0 ? (
          <PostList posts={awards} headingLevel="h2" showSummary={false} contentType="awards" />
        ) : (
          <div className="rounded-3xl border border-dashed border-slate-300 bg-white/70 px-6 py-10 text-center text-sm text-slate-600">
            暂无奖项，先在 `content/awards` 创建一个文件夹并放入 `index.md` 吧。
          </div>
        )}
      </section>
    </SiteLayout>
  );
}

