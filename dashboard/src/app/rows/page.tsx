import Link from "next/link";
import { getEvalRowIds } from "@/lib/db";

export const dynamic = "force-dynamic";

export default async function RowsPage() {
  const ids = await getEvalRowIds();
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Row Inspector</h1>
        <p className="text-gray-400 mt-1">
          Select any of the {ids.length} held-out test rows to compare all 5 model outputs side-by-side.
        </p>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-2">
        {ids.map((id) => (
          <Link
            key={id}
            href={`/rows/${encodeURIComponent(id)}`}
            className="bg-gray-900 border border-gray-800 rounded-lg px-3 py-2 text-sm text-blue-400 hover:border-blue-500 hover:text-blue-300 truncate"
          >
            {id}
          </Link>
        ))}
      </div>
    </div>
  );
}
