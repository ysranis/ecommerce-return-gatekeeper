import Link from "next/link";
import { getEvalRowById } from "@/lib/db";
import RowInspector from "@/components/RowInspector";
import { notFound } from "next/navigation";

export const dynamic = "force-dynamic";

type Props = { params: Promise<{ id: string }> };

export default async function RowDetailPage({ params }: Props) {
  const { id } = await params;
  const seedId = decodeURIComponent(id);
  const rows = await getEvalRowById(seedId);

  if (!rows.length) notFound();

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <Link href="/rows" className="text-sm text-blue-400 hover:text-blue-300">
          ← Back to rows
        </Link>
        <h1 className="text-xl font-bold text-white">{seedId}</h1>
      </div>
      <RowInspector rows={rows} />
    </div>
  );
}
