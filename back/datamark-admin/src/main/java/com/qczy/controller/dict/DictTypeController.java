package com.qczy.controller.dict;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.common.log.BusinessType;
import com.qczy.common.log.Log;
import com.qczy.common.result.Result;
import com.qczy.model.entity.DictTypeEntity;
import com.qczy.model.request.DeleteRequest;
import com.qczy.service.DictTypeService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.util.ObjectUtils;
import org.springframework.web.bind.annotation.*;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/2 15:36
 * @Description:
 */
@RestController
@RequestMapping("/dict/type")
@Api(tags = "字典管理-类型")
public class DictTypeController {



    @Autowired
    private DictTypeService dictTypeService;



    /**
     * 查询字典类型列表
     */
    @GetMapping("/list")
    @ApiOperation("获取带分页带条件查询")
    public Result list(
            @RequestParam Integer page,
            @RequestParam Integer limit,
            @ModelAttribute DictTypeEntity request) {
        Page<DictTypeEntity> pageParam = new Page<>(page, limit);
        IPage<DictTypeEntity> dictTypeEntityList = dictTypeService.selectDictTypeList(pageParam, request);
        return Result.ok(dictTypeEntityList);
    }


    /**
     * 新增保存字典类型
     */

    @PostMapping("/add")
    @ApiOperation("新增")
    @Log(title = "字典管理-类型", businessType = BusinessType.INSERT)
    public Result addSave(@RequestBody  DictTypeEntity dictType) {
        return Result.ok(dictTypeService.insertDictType(dictType));
    }


    /**
     * 修改保存字典类型
     */
    @PostMapping("/edit")
    @ApiOperation("修改")
    @Log(title = "字典管理-类型", businessType = BusinessType.UPDATE)
    public Result editSave(@RequestBody  DictTypeEntity dictType){
        return Result.ok(dictTypeService.updateDictType(dictType));
    }

    /**
     * 删除字典类型
     */

    @DeleteMapping( "/remove")
    @ApiOperation("批量删除")
    @Log(title = "字典管理-类型", businessType = BusinessType.DELETE)
    public Result remove(@RequestBody DeleteRequest request) {
        if (ObjectUtils.isEmpty(request) && request.getIds() == null) {
            return Result.ok("请传入要删除的数据！");
        }

        return Result.ok(dictTypeService.MyDeleteAll(request.getIds()));
    }

    /**
     * 获取所有一级字典类型
     */
    @ApiOperation("获取所有一级字典类型")
    @GetMapping( "/selectDictType")
    public Result selectDictType(){
        return Result.fail(dictTypeService.selectDictType());
    }


}
