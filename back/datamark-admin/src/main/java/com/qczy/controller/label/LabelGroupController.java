package com.qczy.controller.label;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.common.result.Result;
import com.qczy.mapper.TempFileMapper;
import com.qczy.model.entity.FileEntity;
import com.qczy.model.entity.LabelGroupEntity;
import com.qczy.model.entity.TempFileEntity;
import com.qczy.model.request.AssocDataSetRequest;
import com.qczy.model.request.CopyLabelGroupRequest;
import com.qczy.model.request.DeleteRequest;
import com.qczy.model.request.FileIdsRequest;
import com.qczy.service.FileService;
import com.qczy.service.LabelGroupService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.util.ObjectUtils;
import org.springframework.util.StringUtils;
import org.springframework.validation.BindingResult;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import javax.validation.Valid;
import java.util.ArrayList;
import java.util.Objects;

import static com.qczy.common.excel.ExcelVerification.validateExcel;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/19 14:53
 * @Description:
 */
@RestController
@RequestMapping("/label/group")
@Api(tags = "标签组管理")
public class LabelGroupController {


    @Autowired
    private LabelGroupService labelGroupService;
    @Autowired
    private TempFileMapper tempFileMapper;


    /**
     * 查询【请填写功能名称】列表
     */
    @GetMapping("/list")
    @ApiOperation("标签组列表")
    public Result list(@RequestParam Integer page,
                       @RequestParam Integer limit,
                       @ModelAttribute LabelGroupEntity request) {
        Page<LabelGroupEntity> pageParam = new Page<>(page, limit);
        IPage<LabelGroupEntity> labelGroupList = labelGroupService.selectLabelGroupList(pageParam, request);
        return Result.ok(labelGroupList);
    }


    /**
     * 新增保存【请填写功能名称】
     */
    @PostMapping("/add")
    @ApiOperation("新增")
    public Result addSave(@RequestBody LabelGroupEntity labelGroup) {
        if (labelGroupService.isExistLabelGroupName(null, labelGroup.getLabelGroupName())) {
            return Result.fail("标签组名称已存在！");
        }
        if (labelGroupService.isExistLabelEnglishGroupName(null, labelGroup.getEnglishLabelGroupName())) {
            return Result.fail("标签组英文名称已存在！");
        }
        return Result.ok(labelGroupService.insertLabelGroup(labelGroup));
    }


    /**
     * 修改保存【请填写功能名称】
     */
    @PostMapping("/edit")
    @ApiOperation("修改")
    public Result editSave(@RequestBody LabelGroupEntity labelGroup) {
        if (labelGroupService.isExistLabelGroupName(labelGroup.getId(), labelGroup.getLabelGroupName())) {
            return Result.fail("标签组名称已存在！");
        }
        if (labelGroupService.isExistLabelEnglishGroupName(labelGroup.getId(), labelGroup.getEnglishLabelGroupName())) {
            return Result.fail("标签组英文名称已存在！");
        }
        return Result.ok(labelGroupService.updateLabelGroup(labelGroup));
    }

    /**
     * 删除【请填写功能名称】
     */
    @DeleteMapping("/remove")
    @ApiOperation("删除")
    public Result remove(@RequestBody DeleteRequest request) {
        if (ObjectUtils.isEmpty(request) && request.getIds() == null) {
            return Result.ok("请传入要删除的数据！");
        }
        return Result.ok(labelGroupService.deleteLabelGroupByIds(request.getIds()));
    }

    /**
     * 查询所有标签组
     */
    @GetMapping("/selectLabelList")
    @ApiOperation("返回标签组和标签")
    public Result selectLabelList() {
        return Result.ok(labelGroupService.selectLabelList());
    }

    /**
     * 根据数据集id查询标签组id
     */
    @GetMapping("/getSonIdByLabelGroupIds")
    public Result getSonIdByLabelGroupIds(String sonId) {
        if (StringUtils.isEmpty(sonId)) {
            return Result.ok(new ArrayList<Integer>());
        } else {
            return Result.ok(labelGroupService.getSonIdByLabelGroupIds(sonId));
        }
    }


    /**
     * 复制标签组到数据集
     */
    @PostMapping("/copyLabelGroup")
    @ApiOperation("复制标签组到数据集")
    public Result copyLabelGroup(@RequestBody @Valid CopyLabelGroupRequest request, BindingResult bindingResult) {
        if (bindingResult.hasErrors()) {
            return Result.fail(Objects.requireNonNull(bindingResult.getFieldError()).getDefaultMessage());
        }
        return Result.ok(labelGroupService.copyLabelGroup(request));
    }


    /**
     * 关联数据集
     */
    @PostMapping("/assocDataSet")
    @ApiOperation("关联数据集")
    public Result assocDataSet(@RequestBody @Valid AssocDataSetRequest request, BindingResult bindingResult) {
        if (bindingResult.hasErrors()) {
            return Result.fail(Objects.requireNonNull(bindingResult.getFieldError()).getDefaultMessage());
        }
        return Result.ok(labelGroupService.assocDataSet(request));
    }

    /**
     * 查询当前数据集的标签状态 （包含几个标签组，几个自定义标签）
     */
    @GetMapping("/getDataSonLabelStatus")
    @ApiOperation("查询当前数据集的标签状态 （包含几个标签组，几个自定义标签）")
    public Result getDataSonLabelStatus(String sonId) {
        if (StringUtils.isEmpty(sonId)) {
            return Result.fail("数据集id不能为空！");
        }
        return Result.ok(labelGroupService.getDataSonLabelStatus(sonId));
    }

    /**
     * 标签导入
     */
    @PostMapping("/importLabel")
    public Result importLabel(@RequestBody FileIdsRequest request) {
        if (ObjectUtils.isEmpty(request) && request.getFileIds() == null) {
            return Result.fail("文件不能为空！");
        }

        // 校验每个文件
        for (int fileId : request.getFileIds()) {
            TempFileEntity entity = tempFileMapper.selectById(fileId);
            if (entity == null || StringUtils.isEmpty(entity.getFdTempPath())) {
                return Result.fail("文件不能为空！");
            }
            if (!validateExcel(entity.getFdTempPath())) {
                return Result.fail("格式错误，请检查格式后重试！");
            }
            /*String message = validateExcel(entity.getFdTempPath());
            if (!StringUtils.isEmpty(message)) {
                return Result.fail(message);
            }*/
        }

        // 校验通过，准备上传
        int result = labelGroupService.importLabel(request);
        if (result > 0) {
            return Result.ok(1);
        } else {
            return Result.fail("导入失败！");
        }
    }


}
