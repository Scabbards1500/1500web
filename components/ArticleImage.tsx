"use client";

import { useState } from "react";
import ImageModal from "./ImageModal";

type ArticleImageProps = {
  src?: string;
  alt?: string;
};

export default function ArticleImage({ src, alt = "" }: ArticleImageProps) {
  const [isOpen, setIsOpen] = useState(false);
  if (!src) return null;

  return (
    <>
      {/* CSDN 图床有防盗链，不带 referrer 才能正常加载 */}
      <img
        src={src}
        alt={alt}
        loading="lazy"
        referrerPolicy="no-referrer"
        className="my-4 inline-block max-w-full cursor-zoom-in rounded-lg border border-slate-200 transition hover:opacity-90"
        onClick={() => setIsOpen(true)}
      />
      <ImageModal src={src} alt={alt} isOpen={isOpen} onClose={() => setIsOpen(false)} />
    </>
  );
}
