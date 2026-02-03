import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './SemanticTree.css';

interface TreeNode {
  name: string;
  children?: TreeNode[];
  _id: string;
  semantic_path?: string;
  title?: string;
}

interface SemanticTreeProps {
  treeData: any; // This is the 'tree' object from the API response
  selectedTopicName: string;
}

// Helper to check if a node matches the selected topic
const nodeMatchesTopic = (node: TreeNode, topicName: string): boolean => {
  const name = node.title || node.name || '';
  const path = node.semantic_path || '';
  // Match by exact name, or path ending with the topic name
  return name === topicName ||
         path.endsWith('/' + topicName) ||
         path.endsWith('//' + topicName);
};

export const SemanticTree: React.FC<SemanticTreeProps> = ({ treeData, selectedTopicName }) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current || !treeData) return;

    // Fixed container dimensions
    const containerWidth = containerRef.current.clientWidth || 900;
    const containerHeight = 500;
    const margin = { top: 20, right: 120, bottom: 20, left: 120 };

    // Clear previous render
    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .attr('width', containerWidth)
      .attr('height', containerHeight);

    // Create a group for the tree that will be transformed by zoom
    const g = svg.append('g');

    // Build hierarchy
    const root = d3.hierarchy(treeData);

    // Use D3's tree layout with proper node sizing
    // nodeSize([height, width]) - height is vertical spacing, width is horizontal
    const treeLayout = d3.tree<TreeNode>()
      .nodeSize([24, 200])  // 24px vertical per node, 200px horizontal between depths
      .separation((a, b) => {
        // More separation if different parents, or if either has many siblings
        return a.parent === b.parent ? 1 : 1.5;
      });

    // Apply the layout
    treeLayout(root);

    // Draw links (swap x/y for horizontal layout)
    g.selectAll('.link')
      .data(root.links())
      .enter()
      .append('path')
      .attr('class', 'link')
      .attr('d', d3.linkHorizontal<any, any>()
        .x(d => d.y)   // D3 tree's y becomes our horizontal x
        .y(d => d.x)   // D3 tree's x becomes our vertical y
      );

    // Draw nodes (swap x/y for horizontal layout)
    const node = g.selectAll('.node')
      .data(root.descendants())
      .enter()
      .append('g')
      .attr('class', d => {
        const isSelected = nodeMatchesTopic(d.data, selectedTopicName);
        return `node ${d.children ? 'node--internal' : 'node--leaf'} ${isSelected ? 'node--selected' : ''}`;
      })
      .attr('transform', d => `translate(${d.y},${d.x})`)  // y=horizontal, x=vertical
      .attr('data-path', d => d.data.semantic_path || '');

    // Node circles
    node.append('circle')
      .attr('r', d => nodeMatchesTopic(d.data, selectedTopicName) ? 10 : 5)
      .attr('class', d => nodeMatchesTopic(d.data, selectedTopicName) ? 'node-selected' : '');

    // Add a pulsing ring around selected node
    node.filter(d => nodeMatchesTopic(d.data, selectedTopicName))
      .append('circle')
      .attr('r', 16)
      .attr('class', 'node-selected-ring');

    // Node labels
    node.append('text')
      .attr('dy', '0.31em')
      .attr('x', d => d.children ? -12 : 12)
      .attr('text-anchor', d => d.children ? 'end' : 'start')
      .text(d => {
        const title = d.data.title || d.data.name || '';
        return title.length > 40 ? title.substring(0, 40) + '…' : title;
      })
      .attr('class', d => nodeMatchesTopic(d.data, selectedTopicName) ? 'text-selected' : '')
      .append('title')
      .text(d => d.data.title || d.data.name || '');

    // Setup zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.2, 4])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoom);

    // Find the selected node and zoom to it
    console.log('[SemanticTree] Looking for topic:', selectedTopicName);
    const selectedNode = root.descendants().find(d => nodeMatchesTopic(d.data, selectedTopicName));
    console.log('[SemanticTree] Found node:', selectedNode ? selectedNode.data.title || selectedNode.data.name : 'NO');

    if (selectedNode) {
      // In horizontal tree: y is horizontal position, x is vertical position
      const nodeX = selectedNode.y!;
      const nodeY = selectedNode.x!;

      // Calculate transform to center on selected node with zoom
      const scale = 1.5; // Zoom level
      const translateX = containerWidth / 2 - nodeX * scale;
      const translateY = containerHeight / 2 - nodeY * scale;

      // Apply the transform with animation
      svg.transition()
        .duration(750)
        .call(
          zoom.transform,
          d3.zoomIdentity.translate(translateX, translateY).scale(scale)
        );
    } else {
      // No selected node - fit the entire tree in view
      const bounds = (g.node() as SVGGElement)?.getBBox();
      if (bounds) {
        const fullWidth = bounds.width + margin.left + margin.right;
        const fullHeight = bounds.height + margin.top + margin.bottom;
        const scale = Math.min(
          containerWidth / fullWidth,
          containerHeight / fullHeight,
          1
        ) * 0.9;
        const translateX = containerWidth / 2 - (bounds.x + bounds.width / 2) * scale;
        const translateY = containerHeight / 2 - (bounds.y + bounds.height / 2) * scale;

        svg.call(
          zoom.transform,
          d3.zoomIdentity.translate(translateX, translateY).scale(scale)
        );
      }
    }

  }, [treeData, selectedTopicName]);

  if (!treeData) {
    return <div className="p-4 text-slate-500">Loading tree data...</div>;
  }

  return (
    <div
      ref={containerRef}
      className="semantic-tree-container border border-slate-200 rounded-lg bg-slate-50 overflow-hidden"
      style={{ height: '500px' }}
    >
      <svg ref={svgRef}></svg>
      <div className="text-xs text-slate-400 text-center py-2 border-t border-slate-200 bg-white">
        Scroll to zoom · Drag to pan · Selected topic is highlighted
      </div>
    </div>
  );
};
