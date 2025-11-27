import SiteLayout from "@/components/SiteLayout";
import { getPosts } from "@/lib/posts";

const fallbackImages = [
  "https://images.unsplash.com/photo-1498050108023-c5249f4df085?auto=format&fit=crop&w=800&q=80",
  "https://images.unsplash.com/photo-1469474968028-56623f02e42e?auto=format&fit=crop&w=800&q=80",
  "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=800&q=80",
  "https://images.unsplash.com/photo-1487956382158-bb926046304a?auto=format&fit=crop&w=800&q=80",
  "https://images.unsplash.com/photo-1504384308090-c894fdcc538d?auto=format&fit=crop&w=800&q=80",
  "https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&w=800&q=80",
];

export default function GalleryPage() {
  const posts = getPosts();
  const images = posts
    .map((post) => post.image)
    .filter((url): url is string => Boolean(url))
    .concat(fallbackImages)
    .slice(0, 9);

  return (
    <SiteLayout active="Gallery">
      <section className="mx-auto flex max-w-4xl flex-col gap-4 px-6 pt-12 md:px-12">
        <span className="text-xs font-semibold uppercase tracking-[0.35em] text-slate-400">
          Gallery
        </span>
        <h2 className="text-3xl font-semibold text-slate-900">影像与现场</h2>
        <p className="text-sm text-slate-600">
          从实验现场到城市探索，以图像和注释记录灵感。可将 Markdown front matter 中的 `image` 字段用于展示。
        </p>
      </section>

      <section className="mx-auto max-w-4xl px-6 py-12 md:px-12">
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {images.map((image, index) => (
            <figure
              key={`${image}-${index}`}
              className="group relative overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm"
            >
              <img
                src={image}
                alt="Gallery item"
                loading="lazy"
                className="h-56 w-full object-cover transition-transform duration-700 group-hover:scale-105"
              />
              <figcaption className="absolute inset-0 flex items-end bg-gradient-to-t from-black/60 via-transparent to-transparent p-4 text-xs font-semibold uppercase tracking-widest text-white opacity-0 transition-opacity group-hover:opacity-100">
                灵感瞬间
              </figcaption>
            </figure>
          ))}
        </div>
      </section>
    </SiteLayout>
  );
}

