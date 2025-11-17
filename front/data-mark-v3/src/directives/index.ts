import { App, Directive } from "vue";
import throttle from "./modules/throttle";

const directivesList: { [key: string]: Directive } = {
  throttle,
};

const directives = {
  install: function (app: App<Element>) {
    Object.keys(directivesList).forEach(key => {
      app.directive(key, directivesList[key]);
    });
  }
};

export default directives;
