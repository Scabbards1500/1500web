import SiteLayout from "@/components/SiteLayout";
import { getPosts } from "@/lib/posts";
import PublicationContent from "./PublicationContent";

export default function PublicationPage() {
  const posts = getPosts();

  return (
    <SiteLayout active="Research">
      <section className="mx-auto flex max-w-5xl flex-col gap-4 px-6 pt-12 md:px-12">
        <span className="text-xs font-semibold uppercase tracking-[0.35em] text-slate-400">
          Research
        </span>
        <h2 className="text-3xl font-semibold text-slate-900">Research</h2>
      </section>

      <PublicationContent posts={posts} />
    </SiteLayout>
  );
}

