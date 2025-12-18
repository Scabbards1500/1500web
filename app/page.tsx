import Link from "next/link";
import SiteLayout from "@/components/SiteLayout";
import { getAllPosts } from "@/lib/posts";
import HomeContent from "./HomeContent";

export default function Home() {
  const posts = getAllPosts();

  return (
    <SiteLayout active="Homepage">
      <section className="mx-auto flex max-w-5xl flex-col gap-6 px-6 pt-12 md:px-12">
        <span className="text-xs font-semibold uppercase tracking-[0.35em] text-slate-400">
          Welcome
        </span>
        <h2 className="text-4xl font-semibold text-slate-900">Hello, I'm Hanling Wang</h2>
        <p className="text-base text-slate-600">
          I am a researcher interested in building reliable and interpretable AI systems, with a focus on computer vision and large language models. My research experience spans medical image segmentation, multimodal learning, and LLM-based evaluation systems. Moving forward, I aim to explore how structured representations and  reasoning can enhance the reliability of intelligent systems  :-)
        </p>
      </section>

      <HomeContent posts={posts} />
    </SiteLayout>
  );
}
