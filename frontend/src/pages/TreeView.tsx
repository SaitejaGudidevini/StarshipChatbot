import { useEffect, useState, useRef } from 'react';
import * as d3 from 'd3';
import { apiClient } from '../api/client';
import { TreeData, TreeNode } from '../types';
import { Network, ZoomOut, Maximize2 } from 'lucide-react';

export function TreeView() {
    const [treeData, setTreeData] = useState<TreeData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const svgRef = useRef<SVGSVGElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        loadTreeData();
    }, []);

    useEffect(() => {
        if (treeData && svgRef.current && containerRef.current) {
            renderTree();
        }
    }, [treeData]);

    const loadTreeData = async () => {
        try {
            setLoading(true);
            const data = await apiClient.get<TreeData>('/api/tree/data');
            setTreeData(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load tree data');
        } finally {
            setLoading(false);
        }
    };

    const renderTree = () => {
        if (!treeData || !svgRef.current || !containerRef.current) return;

        const container = containerRef.current;
        const width = container.clientWidth;
        const height = 600;

        // Clear previous render
        d3.select(svgRef.current).selectAll('*').remove();

        const svg = d3.select(svgRef.current)
            .attr('width', width)
            .attr('height', height);

        const g = svg.append('g')
            .attr('transform', `translate(100, ${height / 2})`);

        // Zoom behavior
        const zoom = d3.zoom<SVGSVGElement, unknown>()
            .scaleExtent([0.1, 3])
            .on('zoom', (event) => {
                g.attr('transform', event.transform);
            });

        svg.call(zoom);

        // Tree layout
        const treeLayout = d3.tree<TreeNode>().nodeSize([25, 200]);

        // Create hierarchy
        const root = d3.hierarchy(treeData.tree, (d) => d.children);
        (root as any).x0 = height / 2;
        (root as any).y0 = 0;

        // Collapse children initially
        if (root.children) {
            root.children.forEach(collapse);
        }

        let i = 0;
        const duration = 750;

        function collapse(d: d3.HierarchyNode<TreeNode>) {
            if (d.children) {
                (d as any)._children = d.children;
                (d as any)._children.forEach(collapse);
                d.children = undefined;
            }
        }

        function expandAll(d: d3.HierarchyNode<TreeNode>) {
            if ((d as any)._children) {
                d.children = (d as any)._children;
                (d as any)._children = null;
            }
            if (d.children) {
                d.children.forEach(expandAll);
            }
        }

        function update(source: d3.HierarchyNode<TreeNode>) {
            const treeData = treeLayout(root);
            const nodes = treeData.descendants();
            const links = treeData.descendants().slice(1);

            nodes.forEach((d) => { d.y = d.depth * 250; });

            // Update nodes
            const node = g.selectAll<SVGGElement, d3.HierarchyNode<TreeNode>>('g.tree-node')
                .data(nodes, (d: any) => d.id || (d.id = ++i));

            const nodeEnter = node.enter().append('g')
                .attr('class', 'tree-node')
                .attr('transform', () => `translate(${(source as any).y0},${(source as any).x0})`)
                .on('click', (_event, d) => click(d))
                .style('cursor', 'pointer');

            nodeEnter.append('circle')
                .attr('r', 1e-6)
                .style('fill', (d) => getNodeColor(d.data))
                .style('stroke', '#667eea')
                .style('stroke-width', '2px');

            nodeEnter.append('text')
                .attr('dy', '.35em')
                .attr('x', (d) => d.children || (d as any)._children ? -13 : 13)
                .attr('text-anchor', (d) => d.children || (d as any)._children ? 'end' : 'start')
                .text((d) => d.data.title.length > 30 ? d.data.title.substring(0, 30) + '...' : d.data.title)
                .style('fill', '#333')
                .style('font-size', '12px')
                .append('title')
                .text((d) => `${d.data.title}\n${d.data.source_type}\nDepth: ${d.data.depth}\n${d.data.url}`);

            const nodeUpdate = nodeEnter.merge(node as any);

            nodeUpdate.transition()
                .duration(duration)
                .attr('transform', (d) => `translate(${d.y},${d.x})`);

            nodeUpdate.select('circle')
                .attr('r', 6)
                .style('fill', (d) => getNodeColor(d.data));

            const nodeExit = node.exit().transition()
                .duration(duration)
                .attr('transform', () => `translate(${source.y},${source.x})`)
                .remove();

            nodeExit.select('circle').attr('r', 1e-6);
            nodeExit.select('text').style('fill-opacity', 1e-6);

            // Update links
            const link = g.selectAll<SVGPathElement, d3.HierarchyPointLink<TreeNode>>('path.tree-link')
                .data(links, (d: any) => d.id);

            const linkEnter = link.enter().insert('path', 'g')
                .attr('class', 'tree-link')
                .attr('d', () => {
                    const o = { x: (source as any).x0, y: (source as any).y0 };
                    return diagonal(o, o);
                })
                .style('fill', 'none')
                .style('stroke', '#ccc')
                .style('stroke-width', '1.5px');

            linkEnter.merge(link as any).transition()
                .duration(duration)
                .attr('d', (d) => diagonal(d, d.parent!));

            link.exit().transition()
                .duration(duration)
                .attr('d', () => {
                    const o = { x: source.x!, y: source.y! };
                    return diagonal(o, o);
                })
                .remove();

            nodes.forEach((d: any) => {
                d.x0 = d.x;
                d.y0 = d.y;
            });
        }

        function diagonal(s: any, d: any) {
            return `M ${s.y} ${s.x}
              C ${(s.y + d.y) / 2} ${s.x},
                ${(s.y + d.y) / 2} ${d.x},
                ${d.y} ${d.x}`;
        }

        function click(d: d3.HierarchyNode<TreeNode>) {
            if (d.children) {
                (d as any)._children = d.children;
                d.children = undefined;
            } else {
                d.children = (d as any)._children;
                (d as any)._children = null;
            }
            update(d);
        }

        function getNodeColor(node: TreeNode): string {
            if (node.source_type === 'homepage') return '#667eea';
            if (node.source_type === 'heading') return '#f093fb';
            return '#4facfe';
        }

        // Expose expand/collapse functions
        (window as any).treeExpandAll = () => {
            expandAll(root);
            update(root);
        };

        (window as any).treeCollapseAll = () => {
            if (root.children) {
                root.children.forEach(collapse);
            }
            update(root);
        };

        update(root);
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center h-96">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="space-y-6">
                <div>
                    <h2 className="text-3xl font-bold text-slate-900">Tree Visualization</h2>
                    <p className="text-slate-600 mt-1">Interactive hierarchical view of crawled website structure</p>
                </div>
                <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
                    <Network className="w-12 h-12 text-red-400 mx-auto mb-3" />
                    <h3 className="text-lg font-semibold text-red-900 mb-2">No Tree Data Found</h3>
                    <p className="text-red-700">{error}</p>
                    <p className="text-red-600 mt-2">Generate data first using the "Generator" tab.</p>
                </div>
            </div>
        );
    }

    return (
        <div className="space-y-6">
            <div>
                <h2 className="text-3xl font-bold text-slate-900">Tree Visualization</h2>
                <p className="text-slate-600 mt-1">Interactive hierarchical view of crawled website structure</p>
            </div>

            <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-4">
                <div className="flex items-center justify-between mb-4">
                    <div className="flex gap-2">
                        <button
                            onClick={() => (window as any).treeExpandAll?.()}
                            className="flex items-center gap-2 px-3 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors text-sm"
                        >
                            <Maximize2 className="w-4 h-4" />
                            Expand All
                        </button>
                        <button
                            onClick={() => (window as any).treeCollapseAll?.()}
                            className="flex items-center gap-2 px-3 py-2 bg-slate-500 text-white rounded-lg hover:bg-slate-600 transition-colors text-sm"
                        >
                            <ZoomOut className="w-4 h-4" />
                            Collapse All
                        </button>
                    </div>
                    <div className="text-sm text-slate-600">
                        <span className="font-medium">Scroll to zoom</span> • <span className="font-medium">Drag to pan</span> • <span className="font-medium">Click nodes to toggle</span>
                    </div>
                </div>

                <div ref={containerRef} className="border border-slate-200 rounded-lg bg-white overflow-hidden" style={{ height: '600px' }}>
                    <svg ref={svgRef}></svg>
                </div>

                {treeData && (
                    <div className="mt-4 p-4 bg-slate-50 rounded-lg text-sm">
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <div>
                                <span className="font-semibold text-slate-700">Domain:</span>
                                <span className="ml-2 text-slate-600">{treeData.metadata.domain}</span>
                            </div>
                            <div>
                                <span className="font-semibold text-slate-700">Total Nodes:</span>
                                <span className="ml-2 text-slate-600">{treeData.metadata.total_nodes}</span>
                            </div>
                            <div>
                                <span className="font-semibold text-slate-700">Max Depth:</span>
                                <span className="ml-2 text-slate-600">{treeData.metadata.max_depth}</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <div className="flex items-center gap-1">
                                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#667eea' }}></div>
                                    <span className="text-xs text-slate-600">Homepage</span>
                                </div>
                                <div className="flex items-center gap-1">
                                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#f093fb' }}></div>
                                    <span className="text-xs text-slate-600">Headings</span>
                                </div>
                                <div className="flex items-center gap-1">
                                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#4facfe' }}></div>
                                    <span className="text-xs text-slate-600">Links</span>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
