package com.qczy.controller.label;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.common.excel.ExcelVerification;
import com.qczy.common.result.Result;
import com.qczy.model.entity.LabelEntity;
import com.qczy.model.entity.LabelGroupEntity;
import com.qczy.model.request.AddDataSetLabelRequest;
import com.qczy.model.request.DeleteDataSetLabelRequest;
import com.qczy.model.request.DeleteRequest;
import com.qczy.model.request.LabelEntityRequest;
import com.qczy.model.response.DataSetLabelResponse;
import com.qczy.model.response.GroupLabelResponse;
import com.qczy.service.LabelGroupService;
import com.qczy.service.LabelService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.util.ObjectUtils;
import org.springframework.util.StringUtils;
import org.springframework.validation.BindingResult;
import org.springframework.web.bind.annotation.*;

import javax.validation.Valid;
import java.util.List;
import java.util.Objects;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/19 15:25
 * @Description:
 */
@RestController
@RequestMapping("/label")
@Api(tags = "标签管理")
public class LabelController {

    @Autowired
    private LabelService labelService;

    @Autowired
    private LabelGroupService labelGroupService;

    private static final String CODE_REGEX = "^[a-zA-Z0-9/_-]+$";

    /**
     * 查询【请填写功能名称】列表
     */
    @GetMapping("/list")
    public Result list(@RequestParam Integer page,
                       @RequestParam Integer limit,
                       @RequestParam Integer labelGroupId,
                       @ModelAttribute LabelEntity request) {
        Page<LabelEntity> pageParam = new Page<>(page, limit);
        IPage<LabelEntity> labelList = labelService.selectLabelList(pageParam, labelGroupId, request);
        return Result.ok(labelList);
    }


    /**
     * 新增保存【请填写功能名称】
     */

    @PostMapping("/add")
    public Result addSave(@RequestBody LabelEntity label) {
        if (ObjectUtils.isEmpty(label)) {
            return Result.fail("请求对象不能为空！");
        }
        if (StringUtils.isEmpty(label.getLabelGroupId())) {
            return Result.fail("标签组id不能为空！");
        }
        LabelGroupEntity entity = labelGroupService.getById(label.getLabelGroupId());
        if (StringUtils.isEmpty(entity.getEnglishLabelGroupName())) {
            return Result.fail("当前标签组英文标签名称为空，请先新增英文名称后重试！");
        }

        // 不允许标签名在当前标签组存在
        if (labelService.isExistLabelName(label.getLabelGroupId(), label.getLabelName(), null)) {
            return Result.fail("标签名称已存在！");
        }
        // 不允许英文名在当前标签组存在
        if (labelService.isExistEnglishLabelName(label.getLabelGroupId(), label.getEnglishLabelName(), null, entity)) {
            return Result.fail("标签英文名称已存在！");
        }

        // 标签不允许新增英文名称 有 中文 或者 别的字符
        if (!label.getEnglishLabelName().matches(CODE_REGEX)) {
            return Result.fail("标签英文名称格式不正确！");
        }

        return Result.ok(labelService.insertLabel(label, entity));
    }


    /**
     * 修改保存【请填写功能名称】
     */

    @PostMapping("/edit")
    public Result editSave(@RequestBody LabelEntity label) {
        if (ObjectUtils.isEmpty(label)) {
            return Result.fail("请求对象不能为空！");
        }

        LabelGroupEntity entity = null;
        if (label.getLabelGroupId() != null && label.getLabelGroupId() != 0) {
            entity = labelGroupService.getById(label.getLabelGroupId());
            if (StringUtils.isEmpty(entity.getEnglishLabelGroupName())) {
                return Result.fail("当前标签组英文标签名称为空，请先新增英文名称后重试！");
            }

            // 不允许标签名在当前标签组存在
            if (labelService.isExistLabelName(label.getLabelGroupId(), label.getLabelName(), label.getId())) {
                return Result.fail("标签名称已存在！");
            }
            // 不允许英文名在当前标签组存在
            if (labelService.isExistEnglishLabelName(label.getLabelGroupId(), label.getEnglishLabelName(), label.getId(), entity)) {
                return Result.fail("标签英文名称已存在！");
            }
        } else {
            if (labelService.isExistLabelName1(label.getId(), label.getSonId(), label.getLabelName())) {
                return Result.fail("标签名称已存在！");
            }

            if (labelService.isExistEnglishLabelName1(label.getId(), label.getSonId(), label.getEnglishLabelName())) {
                return Result.fail("标签英文名称已存在！");
            }
        }


        // 标签不允许新增英文名称 有 中文 或者 别的字符
        if (!label.getEnglishLabelName().matches(CODE_REGEX)) {
            return Result.fail("标签英文名称格式不正确！");
        }

        return Result.ok(labelService.updateLabel(label, entity));
    }

    /**
     * 删除【请填写功能名称】
     */
    @DeleteMapping("/remove")
    public Result remove(@RequestBody DeleteRequest request) {
        if (ObjectUtils.isEmpty(request) && request.getIds() == null) {
            return Result.ok("请传入要删除的数据！");
        }
        return Result.ok(labelService.deleteLabelByIds(request.getIds()));
    }

    /**
     * 数据集-查询（添加标签组）
     */
    @GetMapping("/selectGroupLabel")
    @ApiOperation("查询所有标签组（不带分页信息）")
    public Result selectGroupLabel() {
        return Result.ok(labelService.selectGroupLabel());
    }

    /**
     * 数据集-查询（添加标签组）
     */
    @GetMapping("/selectGroupLabelPage")
    @ApiOperation("查询所有标签组（带分页信息）")
    public Result selectGroupLabelPage(
            @RequestParam Integer page,
            @RequestParam Integer limit) {
        Page<GroupLabelResponse> pageParam = new Page<>(page, limit);
        IPage<GroupLabelResponse> labelList = labelService.selectGroupLabelPage(pageParam);
        return Result.ok(labelList);
    }

    /**
     * 新增数据集所选的标签
     */
    @PostMapping("/addDataSetAndLabel")
    @ApiOperation("数据集-查询（添加标签组）")
    public Result addDataSetAndLabel(@RequestBody AddDataSetLabelRequest request) {
        if (ObjectUtils.isEmpty(request)) {
            return Result.fail("参数对象不能为空！");
        }
        return Result.ok(labelService.addDataSetAndLabel(request));
    }

    /**
     * 新增标签（不绑定任何标签组）
     */

    @PostMapping("/addSaveLabel")
    @ApiOperation("新增标签（不绑定任何标签组）并且跟数据组关联")
    public Result addSaveLabel(@RequestBody @Valid LabelEntityRequest label, BindingResult bindingResult) {
        if (ObjectUtils.isEmpty(label)) {
            return Result.fail("参数对象不能为空！");
        }
        if (bindingResult.hasErrors()) {
            return Result.fail(Objects.requireNonNull(bindingResult.getFieldError()).getDefaultMessage());
        }
        if (StringUtils.isEmpty(label.getSonId())) {
            return Result.fail("数据集id不能为空！");
        }

        if (labelService.isExistLabelName1(label.getId(), label.getSonId(), label.getLabelName())) {
            return Result.fail("标签名称已存在！");
        }

        if (labelService.isExistEnglishLabelName1(label.getId(), label.getSonId(), label.getEnglishLabelName())) {
            return Result.fail("标签英文名称已存在！");
        }

        // 标签不允许新增英文名称 有 中文 或者 别的字符
        if (!label.getEnglishLabelName().matches(CODE_REGEX)) {
            return Result.fail("标签英文名称格式不正确！");
        }

        return Result.ok(labelService.addSaveLabel(label));
    }


    /**
     * 查询数据集所选的标签
     */
    @GetMapping("/selectDataSetLabel")
    @ApiOperation("查询数据集所选的标签")
    public Result selectDataSetLabel(@RequestParam String sonId) {
        if (StringUtils.isEmpty(sonId)) {
            return Result.fail("数据集id不能为空！");
        }
        List<DataSetLabelResponse> data = labelService.selectDataSetLabel(sonId);
        return Result.ok(data);
    }

    /**
     * 查询数据集所选的标签
     */
    @GetMapping("/selectDataSetLabelPage")
    @ApiOperation("查询数据集所选的标签")
    public Result selectDataSetLabel(
            @RequestParam Integer page,
            @RequestParam Integer limit,
            String sonId,
            String labelName) {
        /*if (StringUtils.isEmpty(sonId)) {
            return Result.fail("数据集id不能为空！");
        }*/
        Page<DataSetLabelResponse> pageParam = new Page<>(page, limit);
        IPage<DataSetLabelResponse> labelList = labelService.selectDataSetLabelPage(pageParam, sonId, labelName);
        return Result.ok(labelList);
    }

    /**
     * 修改数据集所选中的标签
     */
    @PostMapping("/updateDataLabel")
    public Result updateDataLabel(@RequestBody @Valid LabelEntityRequest label, BindingResult bindingResult) {
        if (ObjectUtils.isEmpty(label)) {
            return Result.fail("参数对象不能为空！");
        }
        if (bindingResult.hasErrors()) {
            return Result.fail(Objects.requireNonNull(bindingResult.getFieldError()).getDefaultMessage());
        }
        if (StringUtils.isEmpty(label.getSonId())) {
            return Result.fail("数据集id不能为空！");
        }

        return Result.ok(labelService.updateDataLabel(label));
    }


    /**
     * 删除数据集所所选中的标签
     */

    @DeleteMapping("/deleteDataSetLabel")
    @ApiOperation("删除数据集所所选中的标签")
    public Result deleteDataSetLabel(@RequestBody DeleteDataSetLabelRequest request) {
        if (ObjectUtils.isEmpty(request)) {
            return Result.fail("请求对象不能为空！");
        }

        if (StringUtils.isEmpty(request.getSonId()) || StringUtils.isEmpty(request.getLabelId())) {
            return Result.fail("参数的参数不能为空！");
        }
        return Result.ok(labelService.deleteDataSetLabel(request));
    }


    /**
     * 置顶标签
     */
    @GetMapping("/topUpLabel")
    public Result topUpLabel(@RequestParam Integer labelId, String sonId) {
        if (labelId == null) {
            return Result.fail("标签id不能为空！");
        }
        int result = labelService.topUpLabel(labelId, sonId);
        if (result > 0) {
            return Result.ok(1);
        } else {
            return Result.fail("置顶失败！");
        }
    }


    /**
     * 新增多个（单个）标签
     */
    @PostMapping("/addBatchLabel")
    public Result addBatchLabel(@RequestBody DeleteDataSetLabelRequest deleteDataSetLabelRequest) {
        int i = labelService.deleteDataSetLabelRequest(deleteDataSetLabelRequest);
        if (i > 0) {
            return Result.ok(1);
        } else {
            return Result.fail("标签新增失败！");
        }
    }


}
