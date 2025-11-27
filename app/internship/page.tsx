import Link from "next/link";
import SiteLayout from "@/components/SiteLayout";

const experiences = [
  {
    title: "字节跳动 · 研究实习",
    period: "2024.02 - 2024.08",
    description:
      "负责多模态材料疲劳评估的模型设计与指标体系构建，与产品团队合作完成端到端部署。",
    highlights: ["构建跨模态数据集", "将推理延迟压缩 37%", "落地线上巡检服务"],
  },
  {
    title: "中科院声学所 · 项目合作",
    period: "2023.06 - 2023.12",
    description:
      "参与分布式光纤传感城市桥梁监测项目，搭建数据治理流程并完成实时预警算法。",
    highlights: ["搭建 ETL 管线", "引入自监督特征提取", "实现 28 小时提前预警"],
  },
];

export default function InternshipPage() {
  return (
    <SiteLayout active="Internship">
      <section className="mx-auto flex max-w-4xl flex-col gap-4 px-6 pt-12 md:px-12">
        <span className="text-xs font-semibold uppercase tracking-[0.35em] text-slate-400">
          Internship
        </span>
        <h2 className="text-3xl font-semibold text-slate-900">合作与实践</h2>
        <p className="text-sm text-slate-600">
          将研究经验转化为工程价值，持续在真实环境中迭代验证。以下经验可根据需要增删或替换为真实内容。
        </p>
      </section>

      <section className="mx-auto max-w-4xl px-6 pt-12 md:px-12">
        <div className="flex flex-col gap-8">
          {experiences.map((exp) => (
            <article
              key={exp.title}
              className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm transition-shadow hover:shadow-md"
            >
              <div className="flex flex-col gap-2">
                <h3 className="text-2xl font-semibold text-slate-900">{exp.title}</h3>
                <span className="text-xs font-semibold uppercase tracking-widest text-slate-400">
                  {exp.period}
                </span>
                <p className="text-sm text-slate-600">{exp.description}</p>
              </div>
              <ul className="mt-4 flex flex-wrap gap-2 text-xs uppercase tracking-widest text-slate-500">
                {exp.highlights.map((item) => (
                  <li key={item} className="rounded-full border border-slate-200 px-3 py-1">
                    {item}
                  </li>
                ))}
              </ul>
            </article>
          ))}
        </div>
      </section>

      <section className="mx-auto max-w-4xl px-6 py-16 md:px-12">
        <div className="rounded-3xl border border-dashed border-slate-300 bg-white/70 px-6 py-10 text-center text-sm text-slate-600">
          想要了解更多合作机会？欢迎通过
          <Link href="mailto:hello@example.com" className="font-semibold text-slate-900 underline decoration-slate-300 underline-offset-4">
            邮件
          </Link>
          或社交账号联系我。
        </div>
      </section>
    </SiteLayout>
  );
}

