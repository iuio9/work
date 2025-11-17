package com.qczy.controller.dict;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.common.log.BusinessType;
import com.qczy.common.log.Log;
import com.qczy.common.result.Result;
import com.qczy.model.entity.DictDataEntity;
import com.qczy.model.request.DeleteRequest;
import com.qczy.service.DictDataService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.util.ObjectUtils;
import org.springframework.util.StringUtils;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/5 9:58
 * @Description:
 */
@RestController
@RequestMapping("/dict/data")
@Api(tags = "字典管理-类型-数据")
public class DictDataController {


    @Autowired
    private DictDataService dictDataService;


    /**
     * 查询【请填写功能名称】列表
     */
    @GetMapping("/list")
    @ApiOperation("获取带分页带条件查询")
    public Result list(
            @RequestParam Integer page,
            @RequestParam Integer limit,
            @RequestParam Integer typeId,
            @ModelAttribute DictDataEntity request) {
        if (StringUtils.isEmpty(typeId)) {
            return Result.fail("类型id不能为空！");
        }
        Page<DictDataEntity> pageParam = new Page<>(page, limit);
        IPage<DictDataEntity> dictDataEntityList = dictDataService.selectDictDataList(pageParam, typeId, request);
        return Result.ok(dictDataEntityList);
    }


    /**
     * 新增保存【请填写功能名称】
     */

    @PostMapping("/add")
    @ApiOperation("新增")
    @Log(title = "字典管理-类型-数据", businessType = BusinessType.INSERT)
    public Result addSave(@RequestBody @Validated DictDataEntity dictData) {
        return Result.ok(dictDataService.insertDictData(dictData));
    }


    /**
     * 修改保存【请填写功能名称】
     */

    @PostMapping("/edit")
    @ApiOperation("修改")
    @Log(title = "字典管理-类型-数据", businessType = BusinessType.UPDATE)
    public Result editSave(@RequestBody @Validated DictDataEntity dictData) {
        return Result.ok(dictDataService.updateDictData(dictData));
    }

    /**
     * 删除【请填写功能名称】
     */

    @DeleteMapping("/remove")
    @ApiOperation("批量删除")
    @Log(title = "字典管理-类型-数据", businessType = BusinessType.DELETE)
    public Result remove(@RequestBody DeleteRequest request) {
        if (ObjectUtils.isEmpty(request) && request.getIds() == null) {
            return Result.ok("请传入要删除的数据！");
        }
        if (dictDataService.getFatherAndDataSon(request.getIds()) > 0) {
            return Result.fail("数据集中以绑定此字典数据，请先删除数据集后重试！");
        }
        return Result.ok(dictDataService.deleteDictDataByIds(request.getIds()));
    }


    /**
     * 返回树形结构数据
     */
    @GetMapping("/getDictDataTree")
    @ApiOperation("返回树形结构数据")
    public Result getDictDataTree(@RequestParam Integer typeId) {
        return Result.ok(dictDataService.getDictDataTree(typeId));
    }


    /**
     * 根据多个id批量查询
     */

    @GetMapping("/selectBatchId")
    @ApiOperation("根据批量id查询数据")
    public Result selectBatchId(Integer[] ids) {
        if (ObjectUtils.isEmpty(ids)) {
            return Result.fail("批量id不能为空！");
        }
        return Result.ok(dictDataService.selectBatchId(ids));
    }


    /**
     * 根据最后一级的id查询上级 并最终合并返回字符串
     */
    @GetMapping("/getTreeLevelDict/{dataTypeId}")
    public Result getTreeLevelDict(@PathVariable Integer dataTypeId) {
        return Result.ok(dictDataService.getTreeLevelDict(dataTypeId));
    }

    /**
     *  根据最后一级的id查询上级的id  并最终合并返回数组
     */
    @GetMapping("/getTreeLevelDictIds/{dataTypeId}")
    public Result getTreeLevelDictIds(@PathVariable Integer dataTypeId) {
        return Result.ok(dictDataService.getTreeLevelDictIds(dataTypeId));
    }

}
