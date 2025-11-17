package com.qczy.controller.dataset;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.common.result.Result;
import com.qczy.model.response.DataResponse;
import com.qczy.model.response.DictSetTypeResponse;
import com.qczy.service.DictDataService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/9/24 12:55
 * @Description:
 */
@RestController
@RequestMapping("/data/set/type")
@Api(tags = "数据集-类型列表")
public class DataSetTypeController {


    @Autowired
    private DictDataService dictDataService;


    @GetMapping("/selectDataSetDictList")
    @ApiOperation("返回数据集所有的数据类型和数量")
    public Result selectDataSetDictList() {
        return Result.ok(dictDataService.selectDataSetDictList());
    }


}
