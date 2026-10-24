/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup shdnodes
 *
 * Neural BSDF: a material learned as a small MLP decoder, loaded from a `.nnm` file
 * (doc/neural_rendering/nnm_format.md). Cycles only.
 */

#include "node_shader_util.hh"
#include "node_util.hh"

#include "UI_interface_layout.hh"
#include "UI_resources.hh"

namespace blender {

namespace nodes::node_shader_bsdf_neural_cc {

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Vector>("UV"_ustr)
      .hide_value()
      .description("Where the neural material is looked up (the active UV map if not linked)");
  b.add_input<decl::Vector>("Normal"_ustr).hide_value();
  b.add_output<decl::Shader>("BSDF"_ustr);
}

static void node_shader_buts_neural(ui::Layout &layout, bContext * /*C*/, PointerRNA *ptr)
{
  layout.prop(ptr, "filepath", ui::ITEM_R_SPLIT_EMPTY_NAME, "", ICON_NONE);
}

static void node_shader_init_neural(bNodeTree * /*ntree*/, bNode *node)
{
  node->storage = MEM_new<NodeShaderBsdfNeural>("NodeShaderBsdfNeural");
}

}  // namespace nodes::node_shader_bsdf_neural_cc

/* node type definition */
void register_node_type_sh_bsdf_neural()
{
  namespace file_ns = nodes::node_shader_bsdf_neural_cc;

  static bke::bNodeType ntype;

  sh_node_type_base(&ntype, "ShaderNodeBsdfNeural"_ustr, SH_NODE_BSDF_NEURAL);
  ntype.ui_name = "Neural BSDF";
  ntype.ui_description =
      "Reflection learned by a small neural network and loaded from a .nnm file. Cycles only";
  ntype.enum_name_legacy = "BSDF_NEURAL";
  ntype.nclass = NODE_CLASS_SHADER;
  ntype.add_ui_poll = object_cycles_shader_nodes_poll;
  ntype.declare = file_ns::node_declare;
  ntype.gather_link_search_ops = search_link_ops_for_shader_bsdf_node;
  ntype.initfunc = file_ns::node_shader_init_neural;
  ntype.draw_buttons = file_ns::node_shader_buts_neural;
  bke::node_type_storage(
      ntype, "NodeShaderBsdfNeural", node_free_standard_storage, node_copy_standard_storage);

  bke::node_register_type(ntype);
}

}  // namespace blender
