package com.qczy.controller.datamark;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.common.result.Result;
import com.qczy.model.entity.MarkInfoEntity;
import com.qczy.model.request.DataSonQueryRequest;
import com.qczy.model.response.DataDetailsResponse;
import com.qczy.model.response.DataMarkResponse;
import com.qczy.service.DataMarkService;
import com.qczy.utils.StringUtils;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/22 20:01
 * @Description:
 */
@RestController
@RequestMapping("/data/mark")
@Api(tags = "在线标注部分（接口）")
public class DataMarkController {

    @Autowired
    private DataMarkService dataMarkService;

    /**
     * 在线标注列表
     */
    @GetMapping("/getDataSetMarkList")
    @ApiOperation("在线标注列表")
    public Result getDataSetMarkList(@RequestParam Integer page,
                                     @RequestParam Integer limit,
                                     @ModelAttribute DataSonQueryRequest request) {
        Page<DataMarkResponse> pageParam = new Page<>(page, limit);
        IPage<DataMarkResponse> dataSetMarkList = dataMarkService.getDataSetMarkList(pageParam, request);
        return Result.ok(dataSetMarkList);
    }


    /**
     * 标注（查询图片）
     */
    @GetMapping("/getDataDetails")
    @ApiOperation(" 标注（查询图片）")
    public Result getDataDetails(@RequestParam String sonId, Integer state) {
        if (StringUtils.isEmpty(sonId)) {
            return Result.fail("数据集ID不能为空！");
        }
        return Result.ok(dataMarkService.getDataDetails(sonId, state));

    }


    /**
     * 保存标注信息
     */
    @PostMapping("/addDataMarkInfo")
    @ApiOperation("保存标注信息")
    public Result addDataMarkInfo(@RequestBody MarkInfoEntity markInfoEntity) {
        return Result.ok(dataMarkService.addDataMarkInfo(markInfoEntity));
    }


    @GetMapping("/test01")
    public Result test01(String outSonId){
        dataMarkService.setMarkFileJsonWrite(outSonId);
        return Result.ok();
    }

}
