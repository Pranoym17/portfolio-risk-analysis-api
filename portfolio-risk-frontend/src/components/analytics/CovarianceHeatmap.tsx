import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { chartPalette } from "@/lib/design-system";
import { Chart } from "./Chart";

export function CovarianceHeatmap({
  matrix,
}: {
  matrix: Record<string, Record<string, number>>;
}) {
  const labels = Object.keys(matrix);
  const data = labels.flatMap((row, rowIndex) =>
    labels.map((column, columnIndex) => [columnIndex, rowIndex, Number(matrix[row]?.[column] ?? 0)]),
  );

  return (
    <Card className="rounded-[20px]">
      <CardHeader className="block">
        <div className="text-[11px] uppercase tracking-[0.12em] text-[var(--text-faint)]">Covariance Matrix</div>
        <CardTitle className="mt-2 text-xl tracking-[-0.03em]">Annualized covariance heatmap</CardTitle>
        <CardDescription>Inspect cross-asset relationships that feed the portfolio volatility estimate.</CardDescription>
      </CardHeader>
      <CardContent>
        <Chart
          style={{ height: 340 }}
          option={{
            animation: false,
            grid: { left: 90, right: 24, top: 20, bottom: 70 },
            xAxis: {
              type: "category",
              data: labels,
              axisLabel: { color: chartPalette.axis, rotate: 30 },
              axisLine: { lineStyle: { color: chartPalette.border } },
            },
            yAxis: {
              type: "category",
              data: labels,
              axisLabel: { color: chartPalette.axis },
              axisLine: { lineStyle: { color: chartPalette.border } },
            },
            visualMap: {
              min: Math.min(...data.map((item) => item[2] as number)),
              max: Math.max(...data.map((item) => item[2] as number)),
              calculable: false,
              orient: "horizontal",
              left: "center",
              bottom: 12,
              inRange: {
                color: ["#eff5f7", "#b7d0e0", "#6f9cb8", "#1f5f85"],
              },
              textStyle: { color: chartPalette.axis },
            },
            tooltip: {
              backgroundColor: chartPalette.tooltipBg,
              borderWidth: 0,
              textStyle: { color: chartPalette.tooltipText },
              formatter: (params: { data: [number, number, number] }) => {
                const [x, y, value] = params.data;
                return `${labels[y]} × ${labels[x]}<br/>${value.toFixed(6)}`;
              },
            },
            series: [
              {
                type: "heatmap",
                data,
                label: {
                  show: true,
                  color: "#132033",
                  formatter: (params: { data: [number, number, number] }) => params.data[2].toFixed(3),
                },
                itemStyle: {
                  borderColor: "#ffffff",
                  borderWidth: 1,
                },
              },
            ],
          }}
        />
      </CardContent>
    </Card>
  );
}
