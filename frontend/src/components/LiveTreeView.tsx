import { useEffect, useRef, useMemo } from 'react';
import * as d3 from 'd3';
import { useGenerator } from '../context/GeneratorContext';
import { LiveTreeNode, NodeState } from '../types';
import { Network } from 'lucide-react';

/**
 * LiveTreeView — Real-time D3 tree visualization that shows worker progress.
 *
 * How it works:
 *   1. Reads `liveTree` from GeneratorContext (set once by tree_init SSE event)
 *   2. Reads `nodeStates` (updated on every worker_update / item_completed SSE event)
 *   3. D3 draws circles for each node, colors them by status:
 *        pending    → grey
 *        processing → blue (pulsing)
 *        completed  → green
 *        error      → red
 */

// ─── Color mapping ───
const STATUS_COLORS: Record<string, string> = {
  pending: '#94a3b8',     // slate-400
  processing: '#3b82f6',  // blue-500
  completed: '#22c55e',   // green-500
  error: '#ef4444',       // red-500
};

// ─── Worker color palette (to distinguish workers visually) ───
const WORKER_COLORS = [
  '#3b82f6', // blue
  '#8b5cf6', // violet
  '#f59e0b', // amber
  '#ec4899', // pink
  '#06b6d4', // cyan
  '#10b981', // emerald
  '#f97316', // orange
  '#6366f1', // indigo
];

function getWorkerColor(workerId?: number): string {
  if (workerId === undefined) return STATUS_COLORS.processing;
  return WORKER_COLORS[workerId % WORKER_COLORS.length];
}

// ─── Summary stats from nodeStates ───
function computeStats(nodeStates: Record<string, NodeState>) {
  const values = Object.values(nodeStates);
  return {
    total: values.length,
    pending: values.filter(n => n.status === 'pending').length,
    processing: values.filter(n => n.status === 'processing').length,
    completed: values.filter(n => n.status === 'completed').length,
    error: values.filter(n => n.status === 'error').length,
  };
}

export function LiveTreeView() {
  const { liveTree, nodeStates } = useGenerator();
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const gRef = useRef<d3.Selection<SVGGElement, unknown, null, undefined> | null>(null);
  const initializedRef = useRef(false);

  const stats = useMemo(() => computeStats(nodeStates), [nodeStates]);

  // ─── STEP A: Build tree layout once when liveTree arrives ───
  useEffect(() => {
    if (!liveTree || !svgRef.current || !containerRef.current) return;

    // Only build the tree structure once
    if (initializedRef.current) return;
    initializedRef.current = true;

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = 500;

    // Clear previous
    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height);

    const g = svg.append('g')
      .attr('transform', `translate(80, ${height / 2})`);
    gRef.current = g;

    // Zoom + pan
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 3])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });
    svg.call(zoom);

    // Build D3 hierarchy from our tree
    const root = d3.hierarchy(liveTree, d => d.children);

    // Tree layout
    const treeLayout = d3.tree<LiveTreeNode>().nodeSize([22, 200]);
    treeLayout(root);

    // Draw links (lines between nodes)
    g.selectAll('path.live-link')
      .data(root.links())
      .enter()
      .append('path')
      .attr('class', 'live-link')
      .attr('d', d3.linkHorizontal<d3.HierarchyPointLink<LiveTreeNode>, d3.HierarchyPointNode<LiveTreeNode>>()
        .x(d => d.y)
        .y(d => d.x) as any
      )
      .attr('fill', 'none')
      .attr('stroke', '#e2e8f0')
      .attr('stroke-width', 1.5);

    // Draw nodes (circles + labels)
    const nodes = g.selectAll<SVGGElement, d3.HierarchyPointNode<LiveTreeNode>>('g.live-node')
      .data(root.descendants())
      .enter()
      .append('g')
      .attr('class', 'live-node')
      .attr('transform', d => `translate(${d.y},${d.x})`)
      .attr('data-path', d => d.data.semantic_path);  // for easy lookup

    // Circle for each node
    nodes.append('circle')
      .attr('r', 6)
      .attr('fill', STATUS_COLORS.pending)
      .attr('stroke', '#cbd5e1')
      .attr('stroke-width', 1.5);

    // Label
    nodes.append('text')
      .attr('dy', '0.35em')
      .attr('x', d => (d.children ? -12 : 12))
      .attr('text-anchor', d => (d.children ? 'end' : 'start'))
      .attr('font-size', '11px')
      .attr('fill', '#475569')
      .text(d => {
        const title = d.data.title || '';
        return title.length > 35 ? title.substring(0, 35) + '…' : title;
      })
      .append('title')
      .text(d => `${d.data.title}\n${d.data.semantic_path}`);

    // Center the tree initially
    const bounds = (g.node() as SVGGElement).getBBox();
    const centerX = width / 2 - bounds.x - bounds.width / 2;
    const centerY = height / 2 - bounds.y - bounds.height / 2;
    svg.call(zoom.transform, d3.zoomIdentity.translate(centerX, centerY).scale(0.8));

  }, [liveTree]);

  // ─── STEP B: Update node colors whenever nodeStates changes ───
  useEffect(() => {
    if (!gRef.current) return;

    gRef.current.selectAll<SVGGElement, d3.HierarchyPointNode<LiveTreeNode>>('g.live-node')
      .each(function(d) {
        const path = d.data.semantic_path;
        const state = nodeStates[path];
        const circle = d3.select(this).select('circle');

        if (!state || state.status === 'pending') {
          circle
            .attr('fill', STATUS_COLORS.pending)
            .attr('stroke', '#cbd5e1')
            .attr('stroke-width', 1.5)
            .attr('r', 6);
        } else if (state.status === 'processing') {
          circle
            .attr('fill', getWorkerColor(state.workerId))
            .attr('stroke', getWorkerColor(state.workerId))
            .attr('stroke-width', 3)
            .attr('r', 8);
        } else if (state.status === 'completed') {
          circle
            .attr('fill', STATUS_COLORS.completed)
            .attr('stroke', '#16a34a')
            .attr('stroke-width', 2)
            .attr('r', 7);
        } else if (state.status === 'error') {
          circle
            .attr('fill', STATUS_COLORS.error)
            .attr('stroke', '#dc2626')
            .attr('stroke-width', 2)
            .attr('r', 7);
        }
      });
  }, [nodeStates]);

  // ─── No tree yet ───
  if (!liveTree) {
    return (
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-8 text-center">
        <Network className="w-12 h-12 text-slate-300 mx-auto mb-3" />
        <p className="text-slate-500">Live tree will appear when generation starts</p>
        <p className="text-xs text-slate-400 mt-1">The backend sends the tree structure, then updates nodes in real-time</p>
      </div>
    );
  }

  // ─── Tree is loaded ───
  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200">
      {/* Header with stats */}
      <div className="px-4 py-3 border-b border-slate-200 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Network className="w-4 h-4 text-slate-500" />
          <span className="text-sm font-semibold text-slate-700">Live Tree</span>
        </div>
        <div className="flex items-center gap-3 text-xs">
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: STATUS_COLORS.pending }}></span>
            {stats.pending} pending
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-full animate-pulse" style={{ backgroundColor: STATUS_COLORS.processing }}></span>
            {stats.processing} active
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: STATUS_COLORS.completed }}></span>
            {stats.completed} done
          </span>
          {stats.error > 0 && (
            <span className="flex items-center gap-1">
              <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: STATUS_COLORS.error }}></span>
              {stats.error} failed
            </span>
          )}
        </div>
      </div>

      {/* Tree canvas */}
      <div
        ref={containerRef}
        className="overflow-hidden bg-slate-50"
        style={{ height: '500px' }}
      >
        <svg ref={svgRef}></svg>
      </div>

      {/* Footer hint */}
      <div className="px-4 py-2 border-t border-slate-100 text-xs text-slate-400 text-center">
        Scroll to zoom · Drag to pan
      </div>
    </div>
  );
}
