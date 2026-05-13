// import PostList from "@/components/PostList";
import SiteLayout from "@/components/SiteLayout";
// import { getNotes } from "@/lib/posts";

/** Notes 模块暂时关闭，恢复时取消注释下方实现并恢复导航与 getAllPosts 中的 getNotes */
export default function NotesPage() {
  // const notes = getNotes();

  return (
    <SiteLayout active="Notes">
      <section className="mx-auto flex max-w-5xl flex-col gap-4 px-6 pt-12 md:px-12">
        <span className="text-xs font-semibold uppercase tracking-[0.35em] text-slate-400">
          Notes and inspirations
        </span>
        <h2 className="text-3xl font-semibold text-slate-900">Notes</h2>
        <p className="text-sm text-slate-600">该模块已暂时关闭。</p>
      </section>

      {/*
      <section className="mx-auto max-w-5xl px-6 pt-12 md:px-12">
        {notes.length > 0 ? (
          <PostList posts={notes} headingLevel="h2" showSummary={false} contentType="notes" />
        ) : (
          <div className="rounded-3xl border border-dashed border-slate-300 bg-white/70 px-6 py-10 text-center text-sm text-slate-600">
            Under Construction :-)
          </div>
        )}
      </section>
      */}
    </SiteLayout>
  );
}

