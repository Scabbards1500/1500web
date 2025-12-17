"use client";

import { useState } from "react";
import Link from "next/link";
import type { Post } from "@/lib/posts";
import ImageModal from "./ImageModal";

function renderMarkdown(
  content: string,
  slug: string,
  contentType: string = "research",
  onImageClick: (src: string, alt: string) => void
) {
  const blocks = content
    .split(/\n{2,}/)
    .map((block) => block.trim())
    .filter(Boolean);

  return blocks.map((block, index) => {
    const lines = block.split("\n");
    const bulletList = lines.every((line) => /^[-*]\s+/.test(line));
    const orderedList = lines.every((line) => /^\d+\.\s+/.test(line));

    // 检查是否是图片
    const imageMatch = block.match(/^!\[([^\]]*)\]\(([^)]+)\)$/);
    if (imageMatch) {
      const [, alt, imagePath] = imageMatch;
      // 如果是相对路径，转换为 API 路由
      let imageUrl = imagePath;
      if (!imagePath.startsWith("http://") && !imagePath.startsWith("https://") && !imagePath.startsWith("/")) {
        imageUrl = `/api/content/${contentType}/${slug}/${imagePath}`;
      }
      return (
        <div key={index} className="my-4">
          <img
            src={imageUrl}
            alt={alt || ""}
            className="w-full cursor-pointer rounded-lg border border-slate-200 transition hover:opacity-90"
            loading="lazy"
            onClick={() => onImageClick(imageUrl, alt || "")}
          />
        </div>
      );
    }

    if (bulletList) {
      return (
        <ul key={index} className="list-disc pl-5 text-sm leading-6 text-slate-600">
          {lines.map((line, idx) => (
            <li key={idx}>{line.replace(/^[-*]\s+/, "")}</li>
          ))}
        </ul>
      );
    }

    if (orderedList) {
      return (
        <ol key={index} className="list-decimal pl-5 text-sm leading-6 text-slate-600">
          {lines.map((line, idx) => (
            <li key={idx}>{line.replace(/^\d+\.\s+/, "")}</li>
          ))}
        </ol>
      );
    }

    return (
      <p key={index} className="text-sm leading-7 text-slate-600">
        {block}
      </p>
    );
  });
}

type PostListProps = {
  posts: Post[];
  headingLevel?: "h2" | "h3";
  showSummary?: boolean;
  contentType?: string;
};

function getImageUrl(imagePath: string, slug: string, contentType: string = "research"): string {
  // 如果是完整的 URL，直接返回
  if (imagePath.startsWith("http://") || imagePath.startsWith("https://") || imagePath.startsWith("/")) {
    return imagePath;
  }
  // 相对路径转换为 API 路由
  return `/api/content/${contentType}/${slug}/${imagePath}`;
}

export default function PostList({ posts, headingLevel = "h3", showSummary = true, contentType = "research" }: PostListProps) {
  const [modalImage, setModalImage] = useState<{ src: string; alt: string } | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleImageClick = (src: string, alt: string) => {
    setModalImage({ src, alt });
    setIsModalOpen(true);
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
    // 等待关闭动画完成后再移除组件
    setTimeout(() => {
      setModalImage(null);
    }, 300); // 与动画时间一致
  };

  return (
    <>
      <div className="flex flex-col gap-10">
        {posts.map((post) => (
          <article
            key={post.slug}
            className="flex flex-col gap-6 rounded-3xl border border-slate-200 bg-white p-6 shadow-sm transition-shadow hover:shadow-md md:flex-row"
          >
            <div className="flex flex-1 flex-col gap-4">
              <div className="space-y-2">
                {post.date && (
                  <span className="text-xs font-medium uppercase tracking-widest text-slate-400">
                    {new Date(post.date).toLocaleDateString("en-US", {
                      year: "numeric",
                      month: "short",
                      day: "numeric",
                    })}
                  </span>
                )}
                {headingLevel === "h2" ? (
                  <h2 className="text-2xl font-semibold text-slate-900">{post.title}</h2>
                ) : (
                  <h3 className="text-2xl font-semibold text-slate-900">{post.title}</h3>
                )}
                {post.subtitle && <p className="text-base text-slate-500">{post.subtitle}</p>}
                {post.keywords && post.keywords.length > 0 && (
                  <div className="flex flex-wrap gap-2 pt-1">
                    {post.keywords.map((keyword, idx) => (
                      <span
                        key={idx}
                        className="rounded-full bg-slate-100 px-2.5 py-0.5 text-xs font-medium text-slate-600"
                      >
                        {keyword}
                      </span>
                    ))}
                  </div>
                )}
              </div>

              {showSummary && post.summary && <p className="text-sm text-slate-500">{post.summary}</p>}
              {post.author && <p className="text-sm text-slate-500"> {post.author}</p>}

              <div className="space-y-4">{renderMarkdown(post.content, post.slug, contentType, handleImageClick)}</div>

              {post.links && (
                <div className="flex flex-wrap gap-3 pt-2 text-sm font-medium">
                  {Object.entries(post.links).map(([key, url]) => (
                    <Link
                      key={key}
                      href={url}
                      className="rounded-full border border-slate-300 px-3 py-1 capitalize text-slate-600 transition hover:border-slate-900 hover:text-slate-900"
                    >
                      {key}
                    </Link>
                  ))}
                </div>
              )}
            </div>

            {post.image && (
              <div className="md:w-64">
                <div className="h-48 overflow-hidden rounded-2xl border border-slate-200 bg-slate-100">
                  <img
                    src={getImageUrl(post.image, post.slug, contentType)}
                    alt={post.title}
                    className="h-full w-full cursor-pointer object-cover transition hover:opacity-90"
                    loading="lazy"
                    onClick={() => handleImageClick(getImageUrl(post.image!, post.slug, contentType), post.title)}
                  />
                </div>
              </div>
            )}
          </article>
        ))}
      </div>

      {modalImage && (
        <ImageModal
          src={modalImage.src}
          alt={modalImage.alt}
          isOpen={isModalOpen}
          onClose={handleCloseModal}
        />
      )}
    </>
  );
}

