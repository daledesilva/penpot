(require '[cljs.build.api :as b]
         '[cljs.tagged-literals])

(alter-var-root #'cljs.tagged-literals/*cljs-data-readers*
                assoc 'ux/tr (fn [v] `(uxbox.locales/tr ~v)))

(b/watch (b/inputs "src")
  {:main 'uxbox.core
   :output-to "resources/public/js/main.js"
   :output-dir "resources/public/js"
   :parallel-build false
   :asset-path "/js"
   :optimizations :none
   :pretty-print true
   :language-in  :ecmascript5
   :language-out :ecmascript5
   :verbose true})
