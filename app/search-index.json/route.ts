import { getAllPosts, getCategory, postHref } from "@/lib/posts";
import { toPlainText } from "@/lib/markdown";
import type { SearchDoc } from "@/lib/search";

// 构建时生成一份静态 JSON，搜索在浏览器端完成
export const dynamic = "force-static";

export function GET() {
  const docs: SearchDoc[] = getAllPosts().map((post) => ({
    href: postHref(post),
    title: post.title,
    abstract: post.abstract ?? "",
    category: getCategory(post.category)?.label ?? post.category,
    series: post.series,
    date: post.date,
    text: toPlainText(post.content),
  }));
  return Response.json(docs);
}
