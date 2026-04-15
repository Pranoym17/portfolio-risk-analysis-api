"use client";

import type { CSSProperties, ComponentType } from "react";
import ReactECharts from "echarts-for-react";

type ChartProps = {
  option: object;
  style?: CSSProperties;
};

const ECharts = ReactECharts as unknown as ComponentType<ChartProps>;

export function Chart({ option, style }: ChartProps) {
  return <ECharts option={option} style={style} />;
}
