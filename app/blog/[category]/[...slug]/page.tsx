import Link from "next/link";
import { notFound } from "next/navigation";
import SiteLayout from "@/components/SiteLayout";
import Markdown from "@/components/Markdown";
import TableOfContents from "@/components/TableOfContents";
import { extractToc } from "@/lib/markdown";
import { getAllPosts, getCategory, getPost } from "@/lib/posts";
import "katex/dist/katex.min.css";
import "highlight.js/styles/github-dark.css";

type Params = { params: Promise<{ category: string; slug: string[] }> };

export function generateStaticParams() {
  return getAllPosts().map((post) => ({ category: post.category, slug: post.slug }));
}

async function loadPost({ params }: Params) {
  const { category, slug } = await params;
  // 动态访问时路径段可能仍是编码状态
  return getPost(category, slug.map((s) => decodeURIComponent(s)));
}

export async function generateMetadata(props: Params) {
  const post = await loadPost(props);
  return { title: post ? `${post.title} | Hanling Wang` : "Not Found", description: post?.abstract };
}

export default async function ArticlePage(props: Params) {
  const post = await loadPost(props);
  if (!post) notFound();
  const category = getCategory(post.category)!;

  const toc = (
    <TableOfContents
      items={extractToc(post.content)}
      title={post.title}
      backHref={`/blog/${category.slug}`}
      backLabel={category.label}
    />
  );

  return (
    <SiteLayout active={category.label} sidebar={toc} wide>
      <article className="mx-auto max-w-6xl px-6 pt-12 md:px-10 xl:px-14">
        <nav className="text-xs font-semibold uppercase tracking-[0.25em] text-slate-400">
          <Link href={`/blog/${category.slug}`} className="hover:text-slate-900">
            {category.label}
          </Link>
          {post.series && <span> / {post.series}</span>}
        </nav>
        <h1 className="mt-4 text-3xl font-semibold text-slate-900 md:text-4xl">{post.title}</h1>
        <div className="mt-3 flex flex-wrap items-center gap-3 text-sm text-slate-500">
          {post.date && <time dateTime={post.date}>{post.date.replace(/-/g, ".")}</time>}
          {post.tags?.map((tag) => (
            <span key={tag} className="rounded-full bg-slate-100 px-2.5 py-0.5 text-xs font-medium text-slate-600">
              {tag}
            </span>
          ))}
        </div>
        {post.abstract && (
          <p className="mt-6 rounded-2xl border-l-4 border-sky-300 bg-white px-5 py-4 text-sm leading-6 text-slate-600 shadow-sm">
            {post.abstract}
          </p>
        )}

        <div className="mt-10 rounded-3xl border border-slate-200 bg-white px-6 py-8 shadow-sm md:px-12 md:py-10">
          <Markdown content={post.content} assetDir={post.assetDir} />
        </div>
      </article>
    </SiteLayout>
  );
}
