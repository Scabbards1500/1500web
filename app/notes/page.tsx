import PostList from "@/components/PostList";
import SiteLayout from "@/components/SiteLayout";
import { getNotes } from "@/lib/posts";

export default function NotesPage() {
  const notes = getNotes();

  return (
    <SiteLayout active="Notes">
      <section className="mx-auto flex max-w-5xl flex-col gap-4 px-6 pt-12 md:px-12">
        <span className="text-xs font-semibold uppercase tracking-[0.35em] text-slate-400">
          Notes and inspirations
        </span>
        <h2 className="text-3xl font-semibold text-slate-900">Notes</h2>
        {/* <p className="text-sm text-slate-600">
          这里收集以 Markdown 记录的临时想法、实验心得或阅读笔记。文件位于 `content/notes` 目录，可按文件夹新增与分类。
        </p> */}
      </section>

      <section className="mx-auto max-w-5xl px-6 pt-12 md:px-12">
        {notes.length > 0 ? (
          <PostList posts={notes} headingLevel="h2" showSummary={false} contentType="notes" />
        ) : (
          <div className="rounded-3xl border border-dashed border-slate-300 bg-white/70 px-6 py-10 text-center text-sm text-slate-600">
            Under Construction :-)
          </div>
        )}
      </section>
    </SiteLayout>
  );
}

